from pathlib import Path
import pandas
from sqlalchemy import create_engine


db_path = Path("/data/project/harmonize/data/CAT/s4_r4/final_data")

for t_fname in db_path.glob('*.csv'):
    print(f"Converting {t_fname.as_posix()} to sqlite")
    print("Reading...")
    t_df = pandas.read_csv(t_fname)
    new_fname = t_fname.with_suffix('.sqlite')
    print(f"Storing in {new_fname.as_posix()}")
    engine = create_engine(
        f'sqlite:///{new_fname.as_posix()}', echo=True)
    print("Storing...")
    t_df.to_sql(
        t_fname.stem, con=engine, index=False, if_exists='replace')
    print("Done")
