import sys
import gc
import os
from dataclasses import dataclass

sys.path.append(os.path.join(os.getcwd(), "source", "yue"))

from yue.infer_stage1 import Stage1Pipeline_EXL2, SampleSettings, load_audio_mono, encode_audio, zero_pad_audio_tracks
from yue.codecmanipulator import CodecManipulator
from yue.infer_stage2 import Stage2Pipeline_EXL2
from yue.infer_postprocess import post_process, encode_stage1, encode_stage2
from yue.common import seed_everything
from song import Song
import numpy as np
import torch
from tqdm import tqdm

from omegaconf import OmegaConf
from yue.models.soundstream_hubert_new import SoundStream

@dataclass
class Stage1Config:
    model_path: str = "models/YuE-s1-7B-anneal-en-cot-exl2"
    basic_model_config: str = "./xcodec_mini_infer/final_ckpt/config.yaml"
    resume_path: str = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
    cache_size: int = 6000
    cache_mode: str = "Q4"

@dataclass
class Stage2Config:
    model_path: str = "models/YuE-s2-1B-general-exl2"
    cache_size: int = 7500
    cache_mode: str = "FP16"

@dataclass
class PostProcessConfig:
    basic_model_config: str = "./xcodec_mini_infer/final_ckpt/config.yaml"
    resume_path: str = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
    vocal_decoder_path: str = "./xcodec_mini_infer/decoders/decoder_131000.pth"
    inst_decoder_path: str = "./xcodec_mini_infer/decoders/decoder_151000.pth"
    config_path: str = "./xcodec_mini_infer/decoders/config.yaml"

class GenerationToken:
    NextId: int = 1

    def __init__(self):
        self._active = False
        self._result = True
        self._reason = None

        self._id = GenerationToken.NextId
        GenerationToken.NextId = GenerationToken.NextId + 1
    
    def __call__(self, *args, **kwds):
        return self._active and self._result
    
    def __eq__(self, value):
        return self._id == value

    def id(self):
        return self._id
    
    def start_generation(self):
        self._active = True

    def stop_generation(self
        , result: bool = True
        , reason: str = None):
        self._active = False
        self._result = result
        self._reason = reason

    def active(self):
        return self._active
    
    def result(self):
        return self._result

@dataclass
class GenerationParams:
    token: GenerationToken = None
    max_new_tokens: int = 0
    resume: bool = False
    use_audio_prompt: bool = False
    use_dual_tracks_prompt: bool = False
    prompt_start_time: int = 0
    prompt_end_time: int = 30
    audio_prompt_path: str = None
    instrumental_track_prompt_path: str = None
    vocal_track_prompt_path: str = None
    stage1_guidance_scale: float = 1.5
    stage1_top_p: float = 0.93
    stage1_temperature: float = 1.0
    stage1_repetition_penalty: float = 1.1
    rescale: bool = False
    hq_audio: bool = False
    enable_trim_output: bool = False,
    trim_output_duration: int = 12,
    output_dir: str = "outputs"

class PostProcessor:
    def __init__(self,
                 device: torch.device,
                 config: PostProcessConfig
                 ):
        self._device = device
        self._config = config
        model_config = OmegaConf.load(config.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        self._codec_model = SoundStream(**model_config.generator.config).to(self._device)
        parameter_dict = torch.load(config.resume_path, map_location=self._device, weights_only=True)
        self._codec_model.load_state_dict(parameter_dict["codec_model"])
        self._codec_model.eval()

    def generate(self,
                 input: list,
                 output_name: str,
                 params: GenerationParams) -> tuple:
        return post_process(self._codec_model, self._device, input, output_name, params.output_dir, self._config.config_path, self._config.vocal_decoder_path, self._config.inst_decoder_path, params.rescale, params.hq_audio)

class Generator:
    def __init__(self,
                 cuda_device_idx: int,
                 stage1_config: Stage1Config,
                 stage2_config: Stage1Config,
                 ):
        self._device = torch.device(f"cuda:{cuda_device_idx}" if torch.cuda.is_available() else "cpu")
        self._stage1_config = stage1_config
        self._stage2_config = stage2_config
        self._stage1_pipeline = None
        self._stage2_pipeline = None
        self._post_process = None

    def load_stage1_first(self):
        if self._stage1_pipeline is not None:
            return
                
        self._stage1_pipeline = Stage1Pipeline_EXL2(
            model_path=self._stage1_config.model_path,
            device=self._device,
            basic_model_config=self._stage1_config.basic_model_config,
            resume_path=self._stage1_config.resume_path,
            cache_size=self._stage1_config.cache_size,
            cache_mode=self._stage1_config.cache_mode,
        )

    def load_stage1_second(self):
        self._stage1_pipeline.load_model()

    def unload_stage1(self):
        self._stage1_pipeline.unload_model()
        self._stage1_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

    def load_stage2(self):
        if self._stage2_pipeline is not None:
            return

        self._stage2_pipeline = Stage2Pipeline_EXL2(
            model_path=self._stage2_config.model_path,
            device=self._device,
            cache_size=self._stage2_config.cache_size,
            cache_mode=self._stage2_config.cache_mode,
        )
        
    def unload_stage2(self):
        self._stage2_pipeline.unload_model()
        self._stage2_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

    def load_post_process(self):
        if self._post_process is not None:
            return
        self._post_process = PostProcessor(device=self._device, config=PostProcessConfig())

    def unload_post_process(self):
        self._post_process = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_stage1_start_context(self, song: Song, use_cache: bool):
        head = f"{song.system_prompt()}\n[Genre] {song.genre()}\n{song.lyrics()}"
        context = self._stage1_pipeline.tokenize_text(head) + song.audio_prompt()

        if not use_cache:
            return 0, context

        segment_tokens = []

        for iseg, segment in enumerate(song.segments()):
            tokens = segment.merged_stage1_tracks()

            if not tokens:
                break

            segment_tokens.append(tokens)

        nr_cached_segments = max(len(segment_tokens) - 1, 0)
        
        for iseg in range(nr_cached_segments):
            context = context + self._stage1_pipeline.tokenize_segment_text(str(song[iseg]), segment_tokens[iseg], iseg == 0, True)

        return nr_cached_segments, context

    def get_stage1_prompt(self, song: Song, segidx: int, use_cache: bool):
        segments = song.segments() 
        segment = segments[segidx] if segidx < len(segments) else None
        if segment:
            segment_length = segment.track_length() if segment.track_length() else song.default_track_length()
            tokens = segment.merged_stage1_tracks() if use_cache else []
            return 2 * segment_length - len(tokens), self._stage1_pipeline.tokenize_segment_text(str(segment), tokens, segidx == 0, False)

    def get_stage1_audio_prompt(self,
                                 params: GenerationParams):
        return self._stage1_pipeline.get_audio_prompt_ids(
            use_dual_tracks_prompt = params.use_dual_tracks_prompt,
            vocal_track_prompt_path = params.vocal_track_prompt_path,
            instrumental_track_prompt_path = params.instrumental_track_prompt_path,
            use_audio_prompt = params.use_audio_prompt,
            audio_prompt_path = params.audio_prompt_path,
            prompt_start_time = params.prompt_start_time,
            prompt_end_time = params.prompt_end_time,
        )

    def generate_stage1(self, 
                 input: Song, 
                 params: GenerationParams
                 ) -> Song:

        # Max new tokens or total song length x nr tracks
        max_new_tokens = 2 * (params.max_new_tokens if params.max_new_tokens != None else input.length())
        # Align to two tokens
        max_new_tokens = 2 * ((max_new_tokens + 1) // 2)

        start_segment, start_context = self.get_stage1_start_context(input, params.resume)
        max_segments = len(input.segments())

        prompts = []
        for iseg in range(start_segment, max_segments):
            segment_length, prompt_ids = self.get_stage1_prompt(input, iseg, params.resume)
            prompts.append((segment_length, prompt_ids))
        
        sample_settings = SampleSettings()
        sample_settings.guidance_scale = params.stage1_guidance_scale
        sample_settings.temperature = params.stage1_temperature
        sample_settings.top_p = params.stage1_top_p
        sample_settings.repetition_penalty = params.stage1_repetition_penalty

        raw_output = self._stage1_pipeline.generate(
            generation_token=params.token,
            start_context=start_context,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            sample_settings=sample_settings,
        )

        if not params.token():
            return None

        # Save result
        segments = self._stage1_pipeline.convert_output_to_segments(raw_output, params.use_audio_prompt or params.use_dual_tracks_prompt)
        output = input.clone()
        for iseg, segment in enumerate(segments):
            for itrack, track in enumerate(segment):
                output[iseg].set_track(0, itrack, track)

        return output

    def convert_stage2_output_to_segments(self,
                                   stage2_tracks: list[list[int], list[int]],
                                   segment_lengths: list[int]
                                   ):        
        tracks = [[],[]]

        for itrack, track in enumerate(stage2_tracks):
            start_pos = 0
            for segment_length in segment_lengths:
                end_pos = start_pos + segment_length
                if start_pos != end_pos:
                    tracks[itrack].append(track[start_pos:end_pos])
                    start_pos = end_pos

        return tracks

    def generate_stage2(self, 
                 input: Song, 
                 params: GenerationParams
                 ) -> Song:

        stage1_tracks = input.merge_segments(0)
        stage2_tracks = input.merge_segments(1) if params.resume else [np.empty((0, 8), dtype=np.int64), np.empty((0, 8), dtype=np.int64)]

        cached_tokens = len(stage2_tracks[0])
        cached_batches = cached_tokens // 300
        cached_tokens_end = cached_batches * 300

        for itrack in range(2):
            # Align stage2 to 300 tokens
            track = stage2_tracks[itrack]
            track = track[:cached_tokens_end]
            stage2_tracks[itrack] = track

            # Remove already generated tokens
            track = stage1_tracks[itrack]
            track = track[cached_tokens_end:]
            stage1_tracks[itrack] = track.squeeze().tolist()

        outputs = self._stage2_pipeline.generate(generation_token=params.token, stage1_tracks=stage1_tracks, stage2_tracks=stage2_tracks)

        if not params.token():
            return None

        segment_lengths = [segment.cached_length(0, 0) for segment in input.segments()]

        output_tracks = self.convert_stage2_output_to_segments(stage2_tracks=outputs, segment_lengths=segment_lengths)

        output = input.clone()

        for itrack, _ in enumerate(output_tracks):
            for isegment, segment_data in enumerate(output_tracks[itrack]):
                output[isegment].set_track(1, itrack, segment_data)

        return output

    def post_process(self,
                    input: Song,
                    stage_idx: int,
                    output_name: str,
                    params: GenerationParams) -> tuple:
        encoder = [encode_stage1, encode_stage2]

        if params.enable_trim_output:
            max_duration = params.trim_output_duration * 50
        else:
            max_duration = 0

        tracks = [encoder[stage_idx](track[-max_duration:].flatten().tolist()) for track in input.merge_segments(stage_idx)]
        return self._post_process.generate(input = tracks, output_name = output_name, params = params)

    def set_seed(self, seed: int):
        seed_everything(seed)

def import_audio_tracks(
        cuda_device_idx: int,
        vocal_track_path: str,
        instrumental_track_path: str,
        start_time: int,
        end_time: int):
    
    try:
        device = torch.device(f"cuda:{cuda_device_idx}" if torch.cuda.is_available() else "cpu")
        config = PostProcessConfig()
        model_config = OmegaConf.load(config.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        codec_model = SoundStream(**model_config.generator.config).to(device)
        parameter_dict = torch.load(config.resume_path, map_location=device, weights_only=True)
        codec_model.load_state_dict(parameter_dict["codec_model"])
        codec_model.eval()
        del parameter_dict

        vocal_audio_data = load_audio_mono(vocal_track_path, start_time=start_time, end_time=end_time)
        instrumental_audio_data = load_audio_mono(instrumental_track_path, start_time=start_time, end_time=end_time)

        vocal_audio_data, instrumental_audio_data = zero_pad_audio_tracks(vocal_audio_data, instrumental_audio_data)

        stages = []
        encoders = [(CodecManipulator("xcodec", 0, 1), 0.5), (CodecManipulator("xcodec", 0, 8), 6)]

        for encoder, bw in tqdm(encoders):
            vocals_ids = encode_audio(codec_model, vocal_audio_data, device, target_bw=bw)
            instrumental_ids = encode_audio(codec_model, instrumental_audio_data, device, target_bw=bw)
            
            vocals_ids = np.array(encoder.npy2ids(vocals_ids[0]),dtype=np.int64).reshape(-1, encoder.n_quantizer)
            instrumental_ids = np.array(encoder.npy2ids(instrumental_ids[0]), dtype=np.int64).reshape(-1, encoder.n_quantizer)

            stages.append((vocals_ids, instrumental_ids))

            del vocals_ids
            del instrumental_ids
    finally:
        del vocal_audio_data
        del instrumental_audio_data
        del codec_model
        del encoders
        torch.cuda.empty_cache()
        gc.collect()

    return stages
