import sqlite3
from pathlib import Path
db_path = Path(__file__).parent.parent / "sqlite" / "transcripts.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS transcript_contexts (
#     video_id TEXT NOT NULL,
#     translation_model TEXT NOT NULL,
#     prompt_version TEXT NOT NULL,
#     transcript_hash TEXT NOT NULL,
#     transcript_context TEXT NOT NULL,

#     UNIQUE(video_id, translation_model, prompt_version)
# )
# """)#

# conn.commit()

cursor.execute("SELECT video_id, translation_model, prompt_version, transcript_hash, transcript_context FROM transcript_contexts")
for row in cursor.fetchall():
    print(row)