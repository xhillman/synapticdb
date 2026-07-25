"""Shared hard limits for predictable benchmark resource use."""

FULL_MEMORY_COUNT = 500
FULL_DIRECT_QUERY_COUNT = 25
FULL_ASSOCIATIVE_QUERY_COUNT = 25
SMOKE_MEMORY_COUNT = 50
SMOKE_DIRECT_QUERY_COUNT = 5
SMOKE_ASSOCIATIVE_QUERY_COUNT = 5

MAX_MEMORY_COUNT = FULL_MEMORY_COUNT
# Plausible questions the corpus cannot answer. Only the chained profile
# carries them; they are what lets a promotion change fail rather than win by
# construction on a corpus where every activated node is correct.
FULL_DISTRACTOR_QUERY_COUNT = 12

MAX_QUERY_COUNT = FULL_DIRECT_QUERY_COUNT + FULL_ASSOCIATIVE_QUERY_COUNT + FULL_DISTRACTOR_QUERY_COUNT
MAX_SCHEDULE_COUNT = FULL_MEMORY_COUNT
MAX_WARMUP_COUNT = 50
MAX_SEED_COUNT = 12
# Simulated spans the harness may replay across, bounding both a single run's
# work and how far a graph can be aged in one measurement.
MAX_SIMULATED_DAYS = 3650.0
# Holdout replays behind the diversity gate. Two reproduces the original
# single-repeat probe; more passes distinguish settling from compounding.
MAX_DIVERSITY_PASSES = 10
# Pre-registered before implementing calibration: a randomly chosen correct
# answer must outscore a randomly chosen distractor at least this often, or the
# confidence field cannot be thresholded and should not be claimed.
MIN_CONFIDENCE_AUC = 0.80
MAX_TOP_K = 100
MAX_MANIFEST_FILES = 5
MAX_DATA_FILE_BYTES = 2_000_000
MAX_RECORD_CHARS = 16_384
MAX_RUN_ID_CHARS = 64
MAX_SEED_TEXT_CHARS = 256
MAX_FIXTURE_DIMENSIONS = 4_096
MAX_EMBEDDING_FEATURES = 8_192
MAX_RERANK_CANDIDATES = 80
