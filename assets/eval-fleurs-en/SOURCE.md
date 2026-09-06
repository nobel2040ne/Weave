# Evaluation set: FLEURS

Source: Google FLEURS (`google/fleurs`), split `test`, config `en_us`.
Licence: CC BY 4.0 -- https://creativecommons.org/licenses/by/4.0/
Downloaded: 2026-09-04
Rows: 120

FLEURS is read speech. It is a real, externally comparable benchmark and it
fixes the "no English eval set at all" problem, but it is NOT booth audio: it
has no spontaneous speech, no two-speaker turn-taking, and no room noise. Treat
a score here as a floor, not as evidence the system works at the fair. Record
real booth audio and pass it with `--refs` before trusting any A/B for the demo.

FETCHED WITHOUT `scripts/fetch_fleurs.py`, WHICH CANNOT GET THIS CONFIG.
HuggingFace's datasets-server refuses every `en_us` request -- `/rows` and
`/first-rows` alike -- with `Scan size limit exceeded: attempted to read
401715275 bytes, limit is 300000000`. That is a server-side limit on the
dataset's own parquet layout, not a transient 500 and not something paging round
(`length=1` and deep offsets fail identically). The `ko_kr` config is smaller and
still works, which is why only English is affected.

This set was built by downloading the parquet shard directly --
`https://huggingface.co/api/datasets/google/fleurs/parquet/en_us/test/0.parquet`
(383 MB, 647 rows, one row group) -- and writing out the first 120 `audio.bytes`
blobs with their `transcription` column, which is the same normalized spoken
form `assets/eval-fleurs-ko` carries. Clips are 16 kHz mono, 19.1 minutes total.
Reading the parquet needs `pyarrow`, which is deliberately NOT in
`requirements.txt`: it was installed into a throwaway venv for the conversion so
the pinned runtime environment stayed untouched.
