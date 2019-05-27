from os import path
import shlex
import shutil
import subprocess
import tempfile
import time


def merge_audio_with(original_video_file, target_video_file, time_limit_sec=0):
    dirname = tempfile.gettempdir()
    audio_file = path.join(dirname, 'audio')

    # Get audio codec
    # cmd = (
    #     'ffprobe -show_streams -pretty %s 2>/dev/null | '
    #     'grep codec_type=audio -B 5 | grep codec_name | cut -d "=" -f 2'
    #     % original_video_file
    # )
    # codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    # codec_name = codec_name.strip('\n ')
    # audio_file += ".%s" % codec_name
    audio_file += ".%s" % "aac"

    # Something wrong with original audio codec; use mp3
    # -vn -acodec copy file.<codec-name>
    if not time_limit_sec:
        cmd = 'ffmpeg -y -i %s -vn -acodec copy %s' % (original_video_file, audio_file)
    else:
        cmd = 'ffmpeg -y -i %s -ss 0 -to %s -vn -acodec copy %s' % (original_video_file, time_limit_sec, audio_file)
    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Failed run %s: exit code %s" % (cmd, code))

    # Get video offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    video_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    video_offset = video_offset.strip('\n ')

    # Get audio offset
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=audio -A 28 | grep start_time | cut -d "=" -f 2'
        % original_video_file
    )
    audio_offset = subprocess.check_output(["bash", "-c", cmd]).decode()
    audio_offset = audio_offset.strip('\n ')

    dirname = tempfile.gettempdir()
    video_file = path.join(dirname, 'video')

    # Get video codec
    cmd = (
        'ffprobe -show_streams -pretty %s 2>/dev/null | '
        'grep codec_type=video -B 5 | grep codec_name | cut -d "=" -f 2'
        % original_video_file
    )
    codec_name = subprocess.check_output(["bash", "-c", cmd]).decode()
    codec_name = codec_name.strip('\n ')
    video_file += ".%s" % codec_name

    shutil.copyfile(target_video_file, video_file)
    # subprocess.call(["cp", target_video_file, video_file])
    time.sleep(0.2)

    cmd = (
        'ffmpeg -y -itsoffset %s -i %s '
        '-itsoffset %s -i %s -c copy %s' %
        (video_offset, video_file, audio_offset, audio_file, target_video_file)
    )

    code = subprocess.call(shlex.split(cmd))
    if code != 0:
        raise RuntimeError("Saving video with sound failed: exit code %s" % code)