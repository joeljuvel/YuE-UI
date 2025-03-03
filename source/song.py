import copy
from itertools import takewhile
import re

from einops import rearrange
import numpy as np

def parse_tag(tag_name, tag_data):
    if tag_name == "length":
        if tag_data.endswith('t'):
            return int(tag_data[:-1])
        return int(float(tag_data) * 50)
    return tag_data

# Parse lyrics in segments: name, tags & lyrics
def parse_lyrics(lyrics):
    pattern = r"(#.*?(?=\[))?\[(\w+)\](.*?)(?=\[|\Z|#)"
    raw_segments = re.findall(pattern, lyrics, re.DOTALL)
    tag_pattern = r"#(\w+)(.*?)(?=#|\Z)";

    segments = []
    for segment in raw_segments:
        tag_block, section_name, lyrics = segment
        raw_tags = re.findall(tag_pattern, tag_block, re.DOTALL)        

        tags = dict()
        
        for tag in raw_tags:
            tag_name, tag_data = tag

            tag_name = tag_name.strip().lower()
            tag_data = tag_data.strip()

            try:            
                tags[tag_name] = parse_tag(tag_name, tag_data)
            except:
                print(f"Invalid tag #{tag_name} {tag_data}")
                pass

        segments.append(SongSegment.create(section_name.strip(), tags, lyrics.strip()))

    return segments    
        
class SongSegment:

    def __init__(self):
        self._name = ""
        self._tags = {}
        self._lyrics = ""    
        self._tracks = SongSegment.create_empty_tracks()

    def create_empty_tracks():
        return [[[] for _ in range(Song.NrTracks)] for _ in range(Song.NrStages)]

    def create(name, tags, lyrics):
        segment = SongSegment()
        segment._name = name
        segment._tags = tags
        segment._lyrics = lyrics        
        return segment
    
    def as_str(self):
        return f"[{self._name}]\n{self._lyrics}\n\n"

    def __str__(self):
        return self.as_str()

    def name(self):
        return self._name

    def has_changed(self, other):
         return self._name == other._name and self._tags == other._tags and self._lyrics == other._lyrics

    def cached_length(self, istage, itrack):
        return len(self._tracks[istage][itrack])

    def track_length(self):
        return self._tags.get('length')

    def set_track(self, istage, itrack, track):
        self._tracks[istage][itrack] = track

    def track(self, istage, itrack):
        return self._tracks[istage][itrack]
    
    def merge(self, other):
        self._tracks = copy.deepcopy(other._tracks)

    def merged_stage1_tracks(self):
        """
        Interleave stage 1 track data [V V V] [I I I] -> [V I V I V I]
        """
        codec_ids = np.array([track for track in self._tracks[0]])
        codec_ids = rearrange(codec_ids, "b n -> (n b)", b=Song.NrTracks)
        codec_ids = codec_ids.tolist()
        return codec_ids

class Song():
    NrStages = 2
    NrTracks = 2
    DefaultTrackLength = 1500
    DefaultSystemPrompt = "Generate music from the given lyrics segment by segment."

    def __init__(self):
        self._segments = []
        self._audio_prompt = []
        self._default_track_length = Song.DefaultTrackLength
        self._system_prompt = Song.DefaultSystemPrompt
        self._raw_lyrics = ""
        self._lyrics = ""
        self._genre = ""

    def __str__(self):
        return self._lyrics

    def __iter__(self):
        return self._segments.__iter__()

    def __len__(self):
        return self._segments.__len__()

    def __getitem__(self, index):
        return self._segments[index]
    
    def _prepare_lyrics(self):
        new_segments = parse_lyrics(self._raw_lyrics.strip())

        for iseg, segment in takewhile(lambda e: e[0] < len(self._segments), enumerate(new_segments)):
            segment.merge(self._segments[iseg])

        self._segments = new_segments

        structured_lyrics = [str(seg) for seg in self._segments]
        self._lyrics = "\n".join(structured_lyrics)

    def default_track_length(self):
        return self._default_track_length

    def set_default_track_length(self, track_length):
        self._default_track_length = track_length

    def set_lyrics(self, lyrics_text):
        self._raw_lyrics = lyrics_text
        self._prepare_lyrics()

    def set_system_prompt(self, system_prompt):
        self._system_prompt = system_prompt

    def system_prompt(self):
        return self._system_prompt    

    def set_audio_prompt(self, audio_prompt):
        self._audio_prompt = audio_prompt

    def audio_prompt(self):
        return self._audio_prompt

    def set_genre(self, genre_text):
        self._genre = genre_text

    def genre(self):
        return self._genre
    
    def segments(self):
        return self._segments

    def lyrics(self):
        return self._lyrics

    def length(self):
        length = 0
        for segment in self._segments:
            length = length + (segment.track_length() if segment.track_length() != None else self._default_track_length)
        return length

    def clear_cache(self, istage):
        for segment in self._segments:
            for itrack in range(Song.NrTracks):
                segment.set_track(istage, itrack, [])

    def merge_segments(self, istage):
        tracks = []
        for itrack in range(Song.NrTracks):
            track_full = []
            for segment in self._segments:
                track_full = track_full + segment.track(istage, itrack)
            tracks.append(track_full)
        return tracks

    def length_seconds(self):
        return self.length() / 50

    def clone(self):
        return copy.deepcopy(self)

class GenerationCache:
    def __init__(self, nr_stages: int = 2):
        self._tracks=[[] for i in range(nr_stages)]
        self._segments = []

    def add_segment(self, name, start, end):
        self._segments.append((name, start, end))

    def add_tracks(self, stageidx: int, tracks: list):
        self._tracks[stageidx] = tracks

    def segments(self):
        return self._segments
    
    def track(self, stageidx: int, trackidx: int):
        if stageidx < len(self._tracks) and trackidx < len(self._tracks[stageidx]):
            return self._tracks[stageidx][trackidx]
        return []

    def rewind(self, timems: int):
        iseg = len(self._segments) - 1

        # 20 ms per token at 50 tps
        tokens = timems // 20
        while iseg >= 0:
            name, start, end = self._segments[iseg]
            length = end - start
            if tokens > length:
                tokens = tokens - length
                self._segments = self._segments[:-1]
            else:
                self._segments[iseg] = (name, start, end - tokens)
                return

            iseg = iseg - 1

    def create_from_song(song: Song):
        cache = GenerationCache(nr_stages=Song.NrStages)

        for istage in range(Song.NrStages):
            cache.add_tracks(istage, song.merge_segments(istage))

        dataidx = 0
        for segment in song.segments():
            segment_length = segment.cached_length(0, 0)
            if segment_length == 0:
                continue
            next_dataidx = dataidx + segment_length
            cache.add_segment(segment.name(), dataidx, next_dataidx)
            dataidx = next_dataidx

        return cache

    def transfer_to_song(self, song: Song):
        nr_segments = len(song)
        for isegment, segment in enumerate(self._segments):
            _, start_token, end_token = segment
            if isegment < nr_segments:
                for istage in range(Song.NrStages):
                    for itrack in range(Song.NrTracks):
                        track = self.track(istage, itrack) 

                        elem_size = 8 if istage == 1 else 1
                        start_pos = start_token * elem_size
                        end_pos = end_token * elem_size

                        if start_pos > len(track) or len(track) == 0:
                            continue

                        # Stage 2 might contain less data than Stage 1
                        if end_pos > len(track):
                            end_pos = len(track)
                        
                        song[isegment].set_track(istage, itrack, track[start_pos:end_pos])
    def save(self):
        data = dict()
        data["tracks"] = copy.deepcopy(self._tracks)
        data["segments"] = copy.deepcopy(self._segments)
        return data

    def load(self, data):
        self._tracks = data.get("tracks", self._tracks)
        self._segments = data.get("segments", self._segments)
