import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema
from datetime import datetime

df = pd.read_csv('dataset1.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
if 'index' in df.columns:
    df = df.drop('index', axis=1)

VALID_GENRES = [
    'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime',
    'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat',
    'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical',
    'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal',
    'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass',
    'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk',
    'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grime',
    'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle',
    'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm',
    'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance',
    'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin',
    'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore',
    'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera',
    'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film',
    'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock',
    'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton',
    'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad',
    'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter',
    'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish',
    'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno',
    'trance', 'trip-hop', 'turkish', 'work-out', 'world-music'
]

schema = DataFrameSchema({
    'track_id': Column(str, checks=Check.str_length(22, 22), nullable=False),
    'artists': Column(str, checks=Check.str_length(2, 512), nullable=False),
    'album_name': Column(str, checks=Check.str_length(2, 512), nullable=False),
    'track_name': Column(str, checks=Check.str_length(2, 512), nullable=False),
    'popularity': Column(int, checks=Check.in_range(0, 100), nullable=False),
    'duration_ms': Column(int, checks=[Check.greater_than(0), Check.less_than_or_equal_to(5237760)], nullable=False),
    'explicit': Column(bool, nullable=False),
    'danceability': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'energy': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'key': Column(int, checks=Check.in_range(0, 11), nullable=False),
    'loudness': Column(float, checks=Check.in_range(-45, 5), nullable=False),
    'mode': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'speechiness': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'acousticness': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'instrumentalness': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'liveness': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'valence': Column(float, checks=Check.in_range(0, 1), nullable=False),
    'tempo': Column(float, checks=Check.in_range(0, 256), nullable=False),
    'time_signature': Column(int, checks=Check.in_range(0, 5), nullable=False),
    'track_genre': Column(str, checks=[Check.isin(VALID_GENRES), Check(lambda s: s.nunique() == 114, element_wise=False)], nullable=False)
}, strict=True, coerce=True)

try:
    validated_df = schema.validate(df, lazy=True)
    errors_df = pd.DataFrame()
except pa.errors.SchemaErrors as err:
    errors_df = err.failure_cases
    errors_df['failure_case'] = errors_df['failure_case'].astype(str)

student_name = "Ivanov"
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"{student_name}_{datetime_str}_music_validation_report.parquet"

if not errors_df.empty:
    errors_df.to_parquet(report_filename, engine='pyarrow', index=False, compression='snappy')
else:
    empty_report = pd.DataFrame({
        'message': ['Валидация прошла успешно, ошибок не найдено'],
        'timestamp': [datetime.now()]
    })
    empty_report.to_parquet(report_filename, engine='pyarrow', index=False, compression='snappy')