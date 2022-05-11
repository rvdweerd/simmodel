import ffmpeg
(
    ffmpeg
    .input('results/*.jpg', pattern_type='glob', framerate=2)
    .output('results/test_movie.mp4')
    .run()
)
