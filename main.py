import glob
import os
import random
import re
import time
from datetime import datetime
import math
import librosa
import numpy as np
from pydub import AudioSegment, silence
from collections import defaultdict
from difflib import SequenceMatcher
import argparse

import perlin


def unique_file(filename):
    counter = 1
    file_name_parts = os.path.splitext(filename)
    while os.path.isfile(filename):
        filename = file_name_parts[0] + '_' + str(counter) + file_name_parts[1]
        counter += 1
    return filename


def get_filename_no_ext_no_numbers(file):
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    filename_no_numbers = ''.join(
        [i for i in filename_no_extension if not i.isdigit()])
    return filename_no_numbers


class FingerPrinter():
    def __init__(self):
        self.analysis_files = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob("analysis files/*.json")
        ]

    def join_parts(self, seed, timestamp):
        part_files = glob.glob("output/{0}-{1}-*".format(seed, timestamp))
        part_files.sort(key=lambda f: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', f)])

        # if "P1" not in part_files[0]:
        #     print("No parts detected.")
        #     return

        all_parts = AudioSegment.empty()
        for part_file in part_files:
            all_parts += AudioSegment.from_file(part_file)
        all_parts.export(
            open("output/{0}-{1}.wav".format(seed, timestamp), "wb"),
            format="wav")

        for part_file in part_files:
            os.remove(part_file)

    def run(self,
            source_paths,
            seed=0,
            exclude_paths=[],
            exclude_files=[],
            max_output_length=-1,
            big_chunk_interval=[1, 10, 5],
            big_chunk_length=[500, 5000],
            noise_rate_1=1,
            noise_rate_2=1,
            noise_rate_3=1,
            chunk_duration_range=[5, 300],
            chunk_overlap_range=[0.4, 0.95],
            chunk_skip_range=[300, 5000],
            remove_duplicates=True,
            split_interval=5 * 60,
            file_start_index=0,
            output_mels=False,
            correct_mod_dates=True,
            remove_silence=False,
            max_num_files=-1,
            time_scale=1):
        random.seed(seed)

        noise_offset = random.uniform(0, 10000000)

        noise = perlin.PerlinNoiseFactory(1, octaves=3, unbias=True)

        audio_files = []
        for path in source_paths:
            audio_files += glob.glob(path + "/**/*", recursive=True)

        if max_num_files > 0:
            audio_files = audio_files[:max_num_files]

        filtered_files = []
        for file in audio_files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                valid = True
                for exclude_path in exclude_paths:
                    if exclude_path in file:
                        valid = False
                for exclude_file in exclude_files:
                    if file.endswith(exclude_file):
                        valid = False
                if valid:
                    filtered_files.append(file)
        audio_files = filtered_files

        # get corrected modified dates from other file
        if correct_mod_dates:
            corrected_modified_dates = dict()
            for file in audio_files:
                files_with_same_name = glob.glob(
                    os.path.splitext(file)[0] + ".*")

                if len(files_with_same_name) > 1:
                    files_with_same_name.sort(
                        key=lambda f: os.path.getmtime(f))
                    corrected_modified_dates[file] = os.path.getmtime(
                        files_with_same_name[0])

        # sort by date modified
        def get_date_modified(file):
            return corrected_modified_dates[
                file] if file in corrected_modified_dates else os.path.getmtime(
                    file)

        # correct the timestamps of this folder
        fs = list(
            filter(
                lambda f: "/Volumes/Shared/Projects/Music/Image-Line/Data/Projects/Oldies" in f,
                audio_files))
        for file in fs:
            date = datetime.fromtimestamp(get_date_modified(file))
            if date.day == 21 and date.month == 10 and date.year == 2008:
                date = date.replace(year=2004)
                corrected_modified_dates[file] = date.timestamp()

        audio_files.sort(key=get_date_modified)

        # remove duplicates
        if remove_duplicates:
            deduped_list = []
            filenames = []
            for file in audio_files:
                name = os.path.splitext(os.path.basename(file))[0]
                if name not in filenames:
                    filenames.append(name)
                    deduped_list.append(file)

            audio_files = deduped_list

        def file_interestingness(index):
            if index >= len(audio_files) or index == 0:
                return 1
            else:
                filename = get_filename_no_ext_no_numbers(audio_files[index])
                prev_filename = get_filename_no_ext_no_numbers(
                    audio_files[index - 1])
                return 1 - SequenceMatcher(None, prev_filename,
                                           filename).ratio()

        part_number = 1
        timestamp = int(time.time())

        output_metadata_files_path = "output/{0}-{1}.files.txt".format(
            seed, timestamp)
        output_metadata_times_path = "output/{0}-{1}.times.txt".format(
            seed, timestamp)
        metadata_files_file = open(output_metadata_files_path, "w")
        metadata_times_file = open(output_metadata_times_path, "w")

        num_files_scanned = 1
        fingerprint = AudioSegment.empty()

        total_duration = 0
        all_mels = None
        prev_filename = None

        low_interestingness_time = 0

        for i in range(file_start_index, len(audio_files)):
            file = audio_files[i]
            filename = os.path.basename(file)
            filename_no_extension = os.path.splitext(filename)[0]
            try:
                # open the audio files
                audio_file = AudioSegment.from_file(file)

                # remove silence
                if remove_silence:
                    non_silent_parts = silence.split_on_silence(
                        audio_file, 1000, silence_thresh=-40)
                    non_silent = AudioSegment.empty()
                    for part in non_silent_parts:
                        non_silent += part
                    audio_file = non_silent

                file_duration = audio_file.duration_seconds * 1000

                # skip super short files
                if file_duration < 10:
                    continue

                # calculate interestingness
                interestingness = file_interestingness(i)

                if interestingness == 0:
                    low_interestingness_time += 1

                    if low_interestingness_time > 10:
                        interestingness = 1
                        low_interestingness_time = 0

                    if file_interestingness(i + 1) > 0:
                        interestingness = 1

                # save metadata
                metadata_files_file.write("{0}~{1}\n".format(
                    get_date_modified(file), filename.replace("~", " ")))
                metadata_times_file.write("{0}\n".format(
                    round(total_duration, 2)))

                chunk_start_time = None

                chunk_skip_rate = random.uniform(
                    chunk_skip_range[0], chunk_skip_range[1]) * time_scale

                chunk_start_time = 0

                file_mels = None

                num_big_chunks = 0
                if file_duration > 30000 and file_duration < 240000:
                    num_big_chunks = 1
                else:
                    num_big_chunks = math.floor(file_duration / 120000)
                if num_big_chunks > 5:
                    num_big_chunks = 5

                if interestingness == 0:
                    num_big_chunks = 0

                big_chunk_interval = file_duration / (num_big_chunks + 1)

                next_bigchunk = big_chunk_interval + random.uniform(
                    -10000, 10000)

                while True:
                    noise_env_1 = noise(total_duration * noise_rate_1 +
                                        noise_offset) * 0.5 + 0.5
                    noise_env_2 = noise(total_duration * noise_rate_2 + 1000 +
                                        noise_offset) * 0.5 + 0.5
                    noise_env_3 = noise(total_duration * noise_rate_3 + 2000 +
                                        noise_offset) * 0.5 + 0.5

                    chunk_duration = (noise_env_1 * (chunk_duration_range[1] - chunk_duration_range[0]) + \
                                     chunk_duration_range[0]) * time_scale
                    chunk_overlap_ratio = (noise_env_2 * (chunk_overlap_range[1] - chunk_overlap_range[0]) + \
                                          chunk_overlap_range[0])

                    big_chunk = False

                    # big chunks are longer snippets taken from the track
                    if file_duration > 10000 and chunk_start_time > next_bigchunk and num_big_chunks > 0:
                        chunk_duration = random.uniform(
                            big_chunk_length[0],
                            big_chunk_length[1]) * time_scale
                        next_bigchunk += big_chunk_interval + random.uniform(
                            -10000, 10000)
                        num_big_chunks -= 1
                        big_chunk = True
                        print("Big chunk!")
                    else:
                        big_chunk = False

                    chunk_duration = min(file_duration,
                                         chunk_duration) * time_scale
                    chunk_duration_half = int(chunk_duration / 2)

                    chunk_end_time = chunk_start_time + chunk_duration

                    # if it's reached the end go to the next file
                    if chunk_end_time > file_duration:
                        break

                    chunk = audio_file[chunk_start_time:chunk_end_time]

                    chunk_start_time += chunk_skip_rate

                    # if the chunk is silent skip it
                    if chunk.dBFS == -float("infinity"):
                        print("Silent chunk skipped.")
                        break

                    # analyse
                    if output_mels:
                        data = np.array(list(chunk.get_array_of_samples()))
                        mels = librosa.feature.melspectrogram(
                            y=data, sr=44100, n_mels=100, power=1)
                        mels = np.mean(mels, axis=1)
                        if not file_mels:
                            file_mels = np.reshape(mels, (100, 1))
                        else:
                            file_mels = np.vstack([file_mels, mels])

                    # fade the ends of the chunk
                    fade_time = chunk_duration_half
                    if big_chunk:
                        fade_time = min(chunk_duration_half, 400)
                        chunk_overlap_ratio = 0.2
                    chunk = chunk.fade_in(fade_time).fade_out(fade_time)

                    # attenuate chunk
                    chunk = chunk.apply_gain(-12 if not big_chunk else -6)

                    prev_len = fingerprint.duration_seconds
                    if fingerprint.duration_seconds == 0:
                        fingerprint = chunk
                    else:
                        # add the silence to the end of the fingerprint to make room for the new chunk
                        fingerprint = fingerprint + AudioSegment.silent(
                            duration=chunk_duration *
                            (1 - chunk_overlap_ratio))

                        curr_position = fingerprint.duration_seconds * 1000 - chunk_duration

                        # overlap the chunk with the fingerprint
                        fingerprint = fingerprint.overlay(
                            chunk, position=curr_position)
                    total_duration += fingerprint.duration_seconds - prev_len

                print(file, ":", i + 1, "of", len(audio_files))
                print("Length: {0:.3f} mins".format(total_duration / 60))

                # split and clear at regular intervals
                if fingerprint.duration_seconds > split_interval + 10:
                    source_paths = "output/{0}-{1}-P{2}.wav".format(
                        seed, timestamp, part_number)
                    part_file = fingerprint[:split_interval * 1000]
                    part_file.export(open(source_paths, "wb"), format="wav")

                    fingerprint = fingerprint[split_interval * 1000:]
                    part_number += 1

                # export and quit if max length reached
                if max_output_length > 0 and total_duration > max_output_length:
                    break

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("Failed:", file)

            if output_mels and file_mels is not None:
                if file_mels.shape[1] > 1:
                    file_mels = np.mean(file_mels, axis=1)

                if all_mels is None:
                    all_mels = file_mels
                else:
                    all_mels = np.hstack([all_mels, file_mels])

            num_files_scanned += 1

        source_paths = "output/{0}-{1}-P{2}.wav".format(
            seed, timestamp, part_number)
        fingerprint.export(open(source_paths, "wb"), format="wav")

        # join all the parts
        print("Joining parts...")
        self.join_parts(seed, timestamp)

        print("FINISHED")


parser = argparse.ArgumentParser()
parser.add_argument(
    "paths", nargs="+", help="the folders containing the music to summarise")
parser.add_argument(
    "--exclude_paths", help="folders to exclude from the summary")
parser.add_argument(
    "--exclude_files", help="files to exclude from the summary")
parser.add_argument(
    "--seed",
    default=random.randint(0, 100000000),
    help="set this to get the same result each time")
parser.add_argument(
    "--chunk_duration_range",
    default=[30, 600],
    nargs=2,
    help="the range of the grain size range in ms")
parser.add_argument(
    "--chunk_overlap_range",
    default=[0.9, 0.99],
    nargs=2,
    help="the range of the grain overlap ratio")
parser.add_argument(
    "--chunk_skip_range",
    default=[1000, 10000],
    nargs=2,
    help="the range of the time in ms between each grain")
parser.add_argument(
    "--big_chunk_length",
    default=[100, 4000],
    nargs=2,
    help="the range of the length of the big chunks")
parser.add_argument(
    "--noise_rate_1",
    default=0.05,
    help="the rate at which the grains fluctuate")
parser.add_argument(
    "--noise_rate_2",
    default=0.05,
    help="the rate at which the grains fluctuate")

args = parser.parse_args()

fingerprinter = FingerPrinter()
fingerprinter.run(
    args.paths,
    seed=args.seed,
    exclude_paths=args.exclude_paths,
    exclude_files=args.exclude_files,
    chunk_duration_range=args.chunk_duration_range,
    chunk_overlap_range=args.chunk_overlap_range,
    chunk_skip_range=args.chunk_skip_range,
    big_chunk_length=args.big_chunk_length,
    noise_rate_1=args.noise_rate_1,
    noise_rate_2=args.noise_rate_2,
    split_interval=60,
    time_scale=1)
