"""Generate the frozen, fully synthetic benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from .contracts import (
    FULL_ASSOCIATIVE_QUERY_COUNT,
    FULL_DIRECT_QUERY_COUNT,
    FULL_MEMORY_COUNT,
    MAX_DATA_FILE_BYTES,
    MAX_RECORD_CHARS,
    SMOKE_ASSOCIATIVE_QUERY_COUNT,
    SMOKE_DIRECT_QUERY_COUNT,
    SMOKE_MEMORY_COUNT,
)

ROOT = Path(__file__).resolve().parent / "data"
MEMORY_TYPES = ("factual", "episodic", "procedural", "contextual")
SOURCES = ("incident-log", "runbook", "service-ticket", "field-note")
ChainSpec = tuple[str, str, str, str, str, str, str]


CHAIN_SPECS = (
    (
        "Kestrel Coldchain",
        "Temperature readings from insulated crates disappear for several minutes whenever trucks leave the Alder depot.",
        "Alder depot gateways forward crate telemetry through Zephyr radio modules installed during the spring retrofit.",
        "Zephyr firmware 4.2 suspends radio polling when its eco-sleep flag detects external power loss, even though the gateway battery remains healthy.",
        "The fleet profile now keeps radio polling active during battery operation; packet completeness returned to 99.8 percent on the next twelve routes.",
        "Why do refrigerated shipments lose temperature readings after leaving the Alder depot?",
        "The apparent sensor gap leads through the depot gateway and its radio power policy.",
    ),
    (
        "Lumen Depot",
        "Pickers in zone C scan the correct shelf but receive a neighboring product code roughly once every thirty orders.",
        "Zone C uses compact label printers configured by the Quartz layout profile rather than the standard warehouse profile.",
        "The Quartz profile was exported at 203 DPI while the replacement printers render at 300 DPI, shifting the barcode crop box by four millimeters.",
        "A 300-DPI layout was deployed and the crop coordinates were regenerated; the following 4,800 picks had no adjacent-label substitutions.",
        "What is causing incorrect product codes during picking in zone C?",
        "The scan errors trace from the zone-specific printer profile to a resolution mismatch.",
    ),
    (
        "Willow Clinic",
        "Some afternoon appointments appear twice on practitioner calendars after patients reschedule through the mobile portal.",
        "Portal changes pass through the Saffron scheduling worker before they are written to each practitioner's calendar.",
        "Saffron compares local appointment strings before converting them to UTC, so daylight-saving offsets make the old and new slots look distinct.",
        "The worker now normalizes both timestamps before comparison and a cleanup removed the duplicate slots without changing valid bookings.",
        "Why are rescheduled afternoon visits appearing twice on clinic calendars?",
        "The duplicate booking symptom is connected to the scheduling worker's timestamp comparison order.",
    ),
    (
        "Morrow Orchard",
        "Irrigation valves in the east rows begin their morning cycle almost twenty minutes later each week.",
        "The east controller is the only field unit using the Solace solar timing board introduced in last year's expansion.",
        "Solace board revision B loses oscillator compensation whenever its charge controller enters low-light mode overnight.",
        "Installing the compensated timing package stopped the accumulated drift; valve starts stayed within thirty seconds for six weeks.",
        "Why do the east orchard rows start watering progressively later?",
        "The growing schedule delay leads through the unique solar controller to its uncompensated clock.",
    ),
    (
        "Ember Rail",
        "Cargo status for westbound trains remains unchanged for hours even though station scans are completing normally.",
        "Westbound updates are consumed by the Marigold broker group shared with the overnight reconciliation job.",
        "The reconciliation deployment accidentally reused the live consumer-group name, allowing it to claim and acknowledge status partitions.",
        "Giving reconciliation a separate consumer group restored live status flow and replayed the unprocessed station events.",
        "Why are westbound cargo updates stale despite successful station scans?",
        "The stale status connects through shared message consumption to a colliding consumer-group identity.",
    ),
    (
        "Cedar Museum",
        "Audio guides stop recognizing exhibits in the new west gallery while older galleries continue to work.",
        "The west gallery received a replacement set of Juniper proximity beacons after construction dust damaged the originals.",
        "The replacement beacons rotate identifiers from a new namespace that is absent from the guide application's offline exhibit manifest.",
        "Publishing the new beacon namespace in manifest revision 18 restored exhibit recognition without an application release.",
        "Why do audio guides fail only in the museum's west gallery?",
        "The gallery-specific failure leads through replacement beacons to an outdated offline identifier manifest.",
    ),
    (
        "Harborlight Hotel",
        "Guest key cards issued after noon fail at elevator readers but still open room doors.",
        "Elevator readers validate credentials through the Tidepool access server, while room locks validate cards locally.",
        "Tidepool's midday certificate rollover installed a new leaf certificate without the intermediate authority expected by elevator firmware.",
        "The complete trust chain was added to the access-server bundle and newly issued cards began working at every elevator reader.",
        "Why do newly issued hotel key cards open rooms but not elevators?",
        "The split behavior leads through server-based elevator validation to an incomplete certificate chain.",
    ),
    (
        "Brindle Bakery",
        "Morning bread batches are unusually dense even though mixing times and ingredient weights match the recipe.",
        "Proofing cabinet seven reports humidity through a recently replaced Nimbus sensor cartridge.",
        "The replacement cartridge emits a fractional value while the cabinet controller interprets its input as a percentage, reducing steam output by a factor of one hundred.",
        "A conversion factor was added to the cabinet profile; loaf volume returned to its normal range in the next production run.",
        "Why are morning loaves dense when the recipe and mixing process are unchanged?",
        "The texture problem leads through proofing conditions to a humidity-unit mismatch.",
    ),
    (
        "Silverwake Ferry",
        "Boarding scanners at dock three lose connectivity during the final ten minutes before departure.",
        "Dock three was moved onto the Coral network template when its waiting area was renovated.",
        "The Coral template assigns handheld devices to a guest VLAN whose address lease is shorter than the boarding window.",
        "Applying the transit-device VLAN profile eliminated lease expiration during boarding and stabilized all six scanners.",
        "Why do dock-three boarding scanners disconnect just before departure?",
        "The time-specific disconnect leads through the renovated dock network to an unsuitable lease policy.",
    ),
    (
        "Clearbrook Recycling",
        "The optical sorter recently started sending transparent containers into the mixed-waste lane.",
        "Transparent-item classification uses camera three, which received a replacement lens during scheduled cleaning.",
        "The lens installation reset camera three to automatic exposure, washing out the edge contrast used by the material classifier.",
        "Restoring the fixed exposure profile raised transparent-container classification to 98.6 percent during the validation shift.",
        "Why is the recycling line misclassifying transparent containers?",
        "The material error connects through a serviced camera to an exposure-profile reset.",
    ),
    (
        "Pinebridge University",
        "Room displays across the science building show a blank schedule every Monday morning.",
        "The displays load a weekly schedule manifest from the Rowan campus content service.",
        "Sunday publishing reuses the same manifest filename, and the building cache retains the previous week's expired response until noon Monday.",
        "Versioned manifest filenames and a publish-time purge made new schedules visible immediately on all room displays.",
        "Why are science-building room schedules blank on Monday mornings?",
        "The weekly display failure leads through manifest delivery to a stale building cache.",
    ),
    (
        "Meadow Veterinary Lab",
        "Several sample results cannot be matched to intake records even though technicians scan every tube successfully.",
        "The affected tubes come from partner clinics using six-character identifiers that begin with zero.",
        "The Fern import worker converts scanned identifiers to integers before lookup, permanently removing leading zeros.",
        "Identifier handling was changed to validated strings and the unmatched results were reconciled from the audit queue.",
        "Why are some successfully scanned laboratory samples missing their intake records?",
        "The matching failure leads through a partner identifier format to destructive numeric conversion.",
    ),
    (
        "Granite Survey",
        "Drone elevation maps for the south construction site are consistently about thirty meters above ground-control measurements.",
        "The south project imports positioning data through the Aster terrain conversion profile.",
        "Aster uses ellipsoidal heights while the site's control points use the regional geoid model, creating a nearly constant vertical offset.",
        "Switching the conversion profile to the regional geoid aligned drone surfaces with every surveyed control point.",
        "Why are south-site drone maps vertically offset from surveyed control points?",
        "The constant elevation error leads through the terrain profile to incompatible vertical datums.",
    ),
    (
        "Lantern Theater",
        "Projected subtitles drift behind dialogue as a performance progresses, but they begin each act in sync.",
        "Subtitle timing is driven by the Indigo audio feed installed with the touring production.",
        "Indigo declares a 48-kHz stream while the interface actually delivers 47.952 kHz, causing cumulative timing drift.",
        "Resampling the feed to the declared rate kept subtitles aligned through the full three-hour dress rehearsal.",
        "Why do theater subtitles gradually fall behind even though every act starts synchronized?",
        "The cumulative drift connects through the touring audio feed to a sample-rate discrepancy.",
    ),
    (
        "Hearthside Food Bank",
        "Inventory for several staple items becomes negative during the busiest distribution window.",
        "Mobile checkout tablets send stock decrements through the Clover retry queue when wireless service is congested.",
        "Clover retries timed-out requests without preserving an operation key, so completed decrements can be applied a second time.",
        "Checkout now assigns an immutable operation key to each decrement; duplicate retries are ignored and stock totals remain nonnegative.",
        "Why does food-bank inventory become negative during busy distribution periods?",
        "The stock anomaly leads through congested tablet retries to missing operation idempotency.",
    ),
    (
        "Violet Observatory",
        "The dome closes for high-wind protection on calm evenings several times each month.",
        "Wind data reaches the dome controller through the Boreal weather adapter added for the new roof station.",
        "Boreal labels its readings as meters per second but forwards the station's original knot values unchanged.",
        "Converting knots before publishing reduced false closure alerts while preserving every genuinely windy shutdown.",
        "Why does the observatory dome trigger wind protection during calm weather?",
        "The false safety response leads through the weather adapter to mislabeled units.",
    ),
    (
        "Juniper Gardens",
        "Visitor kiosks return to the default language between pages despite a guest selecting another language.",
        "Language preference is stored by the Petal web client loaded on outdoor kiosk tablets.",
        "Petal writes the preference to a secure cookie, but the outdoor kiosks serve local pages without a secure transport context.",
        "Moving the preference to local kiosk storage preserved language selection throughout each visitor session.",
        "Why do garden kiosks forget a visitor's language selection between pages?",
        "The reset behavior leads through client-side preference storage to an unusable secure cookie.",
    ),
    (
        "Redwood Bicycle Share",
        "The map reports popular docking stations as full even after riders remove several bicycles.",
        "Dock occupancy reaches the map through the Sparrow heartbeat collector.",
        "Sparrow batches fifty heartbeats, but popular stations exceed the collector's flush timeout before a batch fills during uneven traffic.",
        "Reducing the batch size and adding a two-second flush updated station capacity promptly throughout the evening commute.",
        "Why does the bicycle map show full stations after bicycles have been removed?",
        "The stale capacity display leads through heartbeat collection to an oversized batching rule.",
    ),
    (
        "Orchid Ceramics",
        "Kiln four overshoots firing targets by nearly eighty degrees while its control panel reports the requested temperature.",
        "Kiln four received a new temperature probe during the last refractory-lining replacement.",
        "The installed probe is type K, but the controller channel remains configured for the type J probe it replaced.",
        "Changing the controller channel to type K brought reference-cone readings and panel temperatures back into agreement.",
        "Why is kiln four overheating while its display appears correct?",
        "The hidden temperature error leads through a replaced probe to the wrong thermocouple profile.",
    ),
    (
        "Mossland Wildlife Trust",
        "Remote trail cameras now exhaust their batteries in under a week instead of lasting two months.",
        "The affected cameras received the Wren motion package during the winter animal-count survey.",
        "Wren treats swaying vegetation as a fresh trigger on every frame because its debounce state resets after each short recording.",
        "Persisting debounce state across recordings reduced false captures and restored the expected battery lifetime.",
        "Why did trail-camera battery life collapse after the winter survey update?",
        "The power drain leads through the motion package to repeated vegetation triggers.",
    ),
    (
        "Bluehaven Campgrounds",
        "New reservations occasionally disappear from the availability board for ten to fifteen minutes.",
        "The public board reads from the Driftwood reservation replica rather than the booking database.",
        "A nightly campsite-pricing refresh holds one large transaction on Driftwood, delaying replication of concurrent bookings.",
        "Chunking the pricing refresh into small commits kept replication delay below three seconds during subsequent runs.",
        "Why do campground reservations temporarily vanish from the public availability board?",
        "The delayed visibility leads through the read replica to a long-running pricing transaction.",
    ),
    (
        "Glassfin Aquarium",
        "Filtration pumps cycle on and off every few seconds even though tank pressure is stable.",
        "Pump decisions use readings from the Cascade pressure sensor installed beside the main return pipe.",
        "Cascade readings fluctuate by a tiny amount around the exact switch threshold, and the controller has no hysteresis band.",
        "Adding a pressure deadband stopped rapid cycling without changing normal filtration response.",
        "Why are aquarium filtration pumps rapidly cycling under stable pressure?",
        "The mechanical cycling leads through threshold readings to missing controller hysteresis.",
    ),
    (
        "Maple Community Radio",
        "The morning program repeats the same opening songs on consecutive days despite a large approved playlist.",
        "Song selection is handled by the Thistle scheduler restarted during each overnight maintenance window.",
        "Thistle initializes its playlist cursor from the same daily seed whenever no saved cursor is present.",
        "Persisting the cursor before maintenance produced varied openings while retaining the approved rotation rules.",
        "Why does the radio station repeat its opening songs every morning?",
        "The repeated sequence leads through overnight scheduler restarts to an unsaved playlist cursor.",
    ),
    (
        "Sable Textile Cooperative",
        "Indigo dye batches have become visibly lighter even though pigment measurements remain unchanged.",
        "Dye-vat temperature is reported by the Flax probe calibrated during the annual equipment inspection.",
        "The inspection worksheet applied the ambient-water offset twice when programming the Flax probe.",
        "Removing the duplicate offset restored the target shade across three validation batches.",
        "Why are textile dye batches lighter when pigment quantities have not changed?",
        "The shade change leads through vat conditions to a duplicated probe calibration offset.",
    ),
    (
        "Beacon Relief Cooperative",
        "The emergency-supply dashboard shows duplicate pallet counts after scanners reconnect from offline mode.",
        "Offline scans are replayed through the Acorn field queue when a distribution center regains service.",
        "Acorn assigns a new event identifier during replay instead of preserving the scanner's original identifier.",
        "Preserving original event identifiers allowed the inventory service to discard replayed duplicates automatically.",
        "Why do relief-supply pallet counts double after offline scanners reconnect?",
        "The duplicate inventory leads through offline replay to regenerated event identities.",
    ),
)


DIRECT_SPECS = (
    (
        "How often must insulated transport crates receive a seal inspection?",
        "Insulated transport crates require a seal inspection every 90 days; damaged seals must be replaced before the next route.",
    ),
    (
        "How long are grid alert acknowledgements retained?",
        "Grid alert acknowledgements are retained for 14 days in the operations archive.",
    ),
    (
        "What resolution is required for archival map scans?",
        "Archival map scans must be captured at 400 DPI in lossless grayscale format.",
    ),
    (
        "What is the maximum batch size for the parcel rating endpoint?",
        "The parcel rating endpoint accepts at most 75 shipments in one request.",
    ),
    (
        "Which command verifies a greenhouse controller configuration?",
        "Run `grove verify --controller <id>` to validate a greenhouse controller configuration before deployment.",
    ),
    (
        "What is the default campsite reservation hold period?",
        "Unpaid campsite reservations are held for 18 minutes before inventory is released.",
    ),
    (
        "Which header carries a ferry manifest revision?",
        "Ferry manifest clients send the revision in the `X-Manifest-Revision` header.",
    ),
    (
        "What pressure starts the aquarium backup pump?",
        "The aquarium backup pump starts when return-line pressure remains below 1.8 bar for ten seconds.",
    ),
    (
        "How frequently are museum beacon manifests refreshed?",
        "Museum guide tablets refresh their offline beacon manifest every six hours.",
    ),
    (
        "What color space is required for textile shade references?",
        "Textile shade references are stored in the CIELAB color space under D65 illumination.",
    ),
    ("Which port is used by the depot scanner service?", "The depot scanner service listens on TCP port 7443."),
    (
        "How many failed badge attempts lock a hotel staff credential?",
        "A hotel staff credential locks after five failed badge attempts within fifteen minutes.",
    ),
    (
        "What is the wildlife camera's standard recording length?",
        "Wildlife cameras record a twelve-second clip for each accepted motion event.",
    ),
    (
        "Which coordinate system is used for local construction plans?",
        "Local construction plans use the EPSG:26918 projected coordinate system.",
    ),
    (
        "What is the bakery freezer's alert threshold?",
        "The bakery freezer raises an alert above minus 15 degrees Celsius for more than four minutes.",
    ),
    (
        "How should a clinic cancellation reason be represented?",
        "Clinic cancellation reasons use one lowercase code from the published reason-code registry.",
    ),
    (
        "When does the radio station publish its next-day playlist?",
        "The community radio station publishes its next-day playlist at 20:30 local time.",
    ),
    (
        "What is the maximum offline queue size for relief scanners?",
        "Relief scanners retain up to 2,000 offline events before blocking new inventory operations.",
    ),
    (
        "Which checksum protects laboratory result exports?",
        "Laboratory result exports include a SHA-256 checksum in the companion manifest file.",
    ),
    (
        "How long may an orchard valve remain open continuously?",
        "An orchard irrigation valve may remain open continuously for no more than 45 minutes.",
    ),
    (
        "What sample rate is required for theater accessibility audio?",
        "Theater accessibility audio is delivered as 48-kHz, 24-bit PCM.",
    ),
    (
        "Which format is used for bicycle station identifiers?",
        "Bicycle station identifiers use the format `STN-` followed by six decimal digits.",
    ),
    (
        "What is the recycling camera calibration interval?",
        "Recycling-line cameras are calibrated every 30 operating days.",
    ),
    (
        "How long are university room reservations cached?",
        "University room reservations are cached for 120 seconds on hallway displays.",
    ),
    (
        "What is the ceramic kiln's maximum programmed ramp rate?",
        "Ceramic kilns limit programmed heating ramps to 180 degrees Celsius per hour.",
    ),
)


FACILITIES = (
    "Alder Workshop",
    "Bramble Depot",
    "Cobalt Annex",
    "Dovetail Hall",
    "Elm Test Yard",
    "Frostline Store",
    "Garnet Studio",
    "Hazel Pavilion",
    "Ivory Field Office",
    "Jasper Warehouse",
    "Kingfisher Lab",
    "Lilac Terminal",
    "Mica Service Bay",
    "Nectar Greenhouse",
    "Osprey Archive",
)
COMPONENTS = (
    "airflow monitor",
    "battery cabinet",
    "conveyor counter",
    "door controller",
    "environment probe",
    "flow regulator",
    "gateway relay",
    "humidity logger",
    "inventory reader",
    "junction sensor",
    "keypad module",
    "lighting timer",
    "meter bridge",
    "notification panel",
    "occupancy counter",
)


def _memory(
    memory_id: str,
    content: str,
    memory_type: str,
    *,
    source: str,
    session_id: str,
    agent_id: str,
    tags: list[str],
    entities: list[str],
    linked: list[str] | None = None,
    outcome: str | None = None,
    day: int = 0,
) -> dict[str, Any]:
    timestamp = datetime(2028, 1, 1, 9, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "memory_id": memory_id,
        "content": content,
        "memory_type": memory_type,
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "source": source,
        "session_id": session_id,
        "agent_id": agent_id,
        "tags": tags,
        "thread_id": f"thread-{session_id}" if source in {"incident-log", "service-ticket"} else None,
        "entities": entities,
        "outcome": outcome,
        "linked_memory_ids": linked or [],
    }


def _chain_rows(index: int, spec: ChainSpec) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    organization, anchor, bridge_one, bridge_two, target, query, note = spec
    start = (index - 1) * 4 + 1
    ids = [f"mem-{start + step:04d}" for step in range(4)]
    contents = (anchor, bridge_one, bridge_two, target)
    types = ("episodic", "contextual", "contextual", "episodic")
    memories = []
    for step, (memory_id, content, memory_type) in enumerate(zip(ids, contents, types, strict=True)):
        adjacent = (candidate for candidate in (step - 1, step + 1) if 0 <= candidate < len(ids))
        memories.append(
            _memory(
                memory_id,
                content,
                memory_type,
                source=SOURCES[step],
                session_id=f"chain-{index:02d}",
                agent_id=f"agent-{(index - 1) % 5 + 1:02d}",
                tags=["synthetic", "chain", organization.lower().replace(" ", "-")],
                entities=[organization],
                linked=[ids[candidate] for candidate in adjacent],
                outcome="resolved" if step == 3 else None,
                day=index * 3 + step,
            )
        )
    query_id = f"Q-{FULL_DIRECT_QUERY_COUNT + index:03d}"
    query_row = {
        "query_id": query_id,
        "label": "associative",
        "text": query,
        "expected_relevant_node_ids": [ids[3]],
        "required_intermediate_node_ids": [ids[1], ids[2]],
        "reviewer_note": note,
    }
    section = [
        f"## {query_id}",
        "",
        f"**Query:** {query}",
        "",
        "**Chain:**",
        "",
        f"- Anchor `{ids[0]}`: {anchor}",
        f"- Intermediate `{ids[1]}`: {bridge_one}",
        f"- Intermediate `{ids[2]}`: {bridge_two}",
        f"- **Target** `{ids[3]}`: {target}",
        "",
        f"*Note: {note}*",
        "",
        "---",
        "",
    ]
    return memories, query_row, section


def _build_associative_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    memories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    sections = [
        "# Synthetic Associative Query Chains",
        "",
        "All organizations, systems, incidents, and values in this dataset are fictional.",
        "",
    ]
    for index, spec in enumerate(CHAIN_SPECS, 1):
        chain_memories, query, section = _chain_rows(index, spec)
        memories.extend(chain_memories)
        queries.append(query)
        sections.extend(section)
    return memories, queries, sections


def _build_direct_rows(start: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for index, (question, answer) in enumerate(DIRECT_SPECS, 1):
        memory_id = f"mem-{start + index - 1:04d}"
        memory_type = "procedural" if answer.startswith("Run ") else "factual"
        memories.append(
            _memory(
                memory_id,
                answer,
                memory_type,
                source=SOURCES[index % len(SOURCES)],
                session_id=f"reference-{index:02d}",
                agent_id=f"agent-{index % 5 + 1:02d}",
                tags=["synthetic", "reference", memory_type],
                entities=["Fictional Operations Handbook"],
                day=100 + index,
            )
        )
        queries.append(
            {
                "query_id": f"Q-{index:03d}",
                "label": "direct",
                "text": question,
                "expected_relevant_node_ids": [memory_id],
                "required_intermediate_node_ids": [],
                "reviewer_note": "Direct lookup of a fictional operating rule.",
            }
        )
    return memories, queries


def _background_content(memory_type: str, facility: str, component: str, code: str, index: int) -> str:
    if memory_type == "factual":
        interval = (index % 11) + 2
        return f"{facility} records {component} readings under reference {code}; the standard review interval is {interval} hours."
    if memory_type == "episodic":
        return f"During synthetic drill {code}, the {component} at {facility} completed its verification cycle with no service interruption."
    if memory_type == "procedural":
        return f"To verify the {component} at {facility}, open checklist {code}, capture one baseline reading, and record the final indicator state."
    return f"At {facility}, {component} alerts are informational during exercise window {code} unless two consecutive readings exceed the local limit."


def _append_background_rows(memories: list[dict[str, Any]]) -> None:
    type_counts = Counter(memory["memory_type"] for memory in memories)
    first_index = len(memories) + 1
    for index in range(first_index, FULL_MEMORY_COUNT + 1):
        memory_type = min(MEMORY_TYPES, key=lambda item: (type_counts[item], MEMORY_TYPES.index(item)))
        facility = FACILITIES[index % len(FACILITIES)]
        component = COMPONENTS[(index * 7) % len(COMPONENTS)]
        code = f"REF-{index:04d}"
        memories.append(
            _memory(
                f"mem-{index:04d}",
                _background_content(memory_type, facility, component, code, index),
                memory_type,
                source=SOURCES[index % len(SOURCES)],
                session_id=f"filler-{(index - 126) // 3:03d}",
                agent_id=f"agent-{index % 5 + 1:02d}",
                tags=["synthetic", "background", memory_type],
                entities=[facility, component],
                day=150 + index,
            )
        )
        type_counts[memory_type] += 1


def _build_schedule(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not 1 <= len(memories) <= FULL_MEMORY_COUNT:
        raise RuntimeError(f"schedule requires 1 to {FULL_MEMORY_COUNT} memories")
    schedule: list[dict[str, Any]] = []
    offset = 0
    for index, memory in enumerate(memories):
        if index > 0:
            previous = memories[index - 1]
            linked = memory["memory_id"] in previous["linked_memory_ids"]
            offset += 120 if linked else 900
        schedule.append({"memory_id": memory["memory_id"], "ingest_offset_seconds": offset})
    return schedule


def _warmup_rows() -> list[dict[str, Any]]:
    return [
        {"text": "How should an airflow monitor baseline be recorded during a drill?", "positive": True},
        {"text": "Where are battery cabinet review intervals documented?", "positive": True},
        {"text": "What should be captured when checking a conveyor counter?", "positive": True},
        {"text": "Which checklist is used for a door controller verification?", "positive": True},
        {"text": "When is an environment probe alert informational?", "positive": True},
        {"text": "What was the fictional lunar terminal's catering budget?", "positive": False},
    ]


def build_full() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    memories, associative_queries, sections = _build_associative_rows()
    direct_memories, direct_queries = _build_direct_rows(len(memories) + 1)
    memories.extend(direct_memories)
    _append_background_rows(memories)
    queries = direct_queries + associative_queries
    schedule = _build_schedule(memories)
    warmup = _warmup_rows()
    expected = (FULL_MEMORY_COUNT, FULL_DIRECT_QUERY_COUNT, FULL_ASSOCIATIVE_QUERY_COUNT)
    _validate_generated(memories, queries, schedule, warmup, expected)
    return memories, queries, schedule, warmup, "\n".join(sections)


def _validate_generated(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    warmup: list[dict[str, Any]],
    expected_counts: tuple[int, int, int],
) -> None:
    memory_ids = [memory["memory_id"] for memory in memories]
    known = set(memory_ids)
    by_id = {memory["memory_id"]: memory for memory in memories}
    counts = (
        len(memories),
        sum(query["label"] == "direct" for query in queries),
        sum(query["label"] == "associative" for query in queries),
    )
    if counts != expected_counts or len(known) != len(memory_ids):
        raise RuntimeError(f"generated dataset count or ID invariant failed: {counts}")
    if len(schedule) != len(memories) or [row["memory_id"] for row in schedule] != memory_ids:
        raise RuntimeError("generated schedule does not match memory order")
    offsets = [row["ingest_offset_seconds"] for row in schedule]
    if offsets[0] != 0 or any(left >= right for left, right in pairwise(offsets)):
        raise RuntimeError("generated schedule offsets must increase from zero")
    if any("synthetic" not in memory["tags"] for memory in memories):
        raise RuntimeError("every generated memory must be marked synthetic")
    links = [
        (memory_id, linked)
        for memory_id, memory in zip(memory_ids, memories, strict=True)
        for linked in memory["linked_memory_ids"]
    ]
    if any(linked not in known or memory_id not in by_id[linked]["linked_memory_ids"] for memory_id, linked in links):
        raise RuntimeError("generated memory links must be known and symmetric")
    if any(
        not set(query["expected_relevant_node_ids"] + query["required_intermediate_node_ids"]) <= known
        for query in queries
    ):
        raise RuntimeError("generated query annotations reference unknown memories")
    holdout = {query["text"].casefold() for query in queries}
    if not warmup or any(row["text"].casefold() in holdout for row in warmup):
        raise RuntimeError("generated warmup must be non-empty and disjoint")


def _jsonl(rows: list[dict[str, Any]]) -> str:
    if not 1 <= len(rows) <= FULL_MEMORY_COUNT:
        raise RuntimeError(f"JSONL generation requires 1 to {FULL_MEMORY_COUNT} rows")
    lines: list[str] = []
    total_bytes = 0
    for row in rows:
        line = json.dumps(row, sort_keys=True) + "\n"
        if len(line) > MAX_RECORD_CHARS:
            raise RuntimeError(f"generated record exceeds {MAX_RECORD_CHARS} characters")
        total_bytes += len(line.encode())
        if total_bytes > MAX_DATA_FILE_BYTES:
            raise RuntimeError(f"generated JSONL exceeds {MAX_DATA_FILE_BYTES} bytes")
        lines.append(line)
    return "".join(lines)


def _write_profile(
    root: Path,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    warmup: list[dict[str, Any]],
    chains: str | None = None,
) -> None:
    payloads = {
        "memories.jsonl": _jsonl(memories),
        "queries.jsonl": _jsonl(queries),
        "schedule.jsonl": _jsonl(schedule),
        "warmup.jsonl": _jsonl(warmup),
    }
    if chains is not None:
        payloads["chains.md"] = chains.rstrip() + "\n"
    if any(len(payload.encode()) > MAX_DATA_FILE_BYTES for payload in payloads.values()):
        raise RuntimeError(f"generated profile exceeds the {MAX_DATA_FILE_BYTES}-byte file limit")
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_text_exact(root / name, payload)
    manifest = {
        "profile": root.name,
        "generator": {"name": "bench.generate_dataset", "version": 1},
        "counts": {
            "memories": len(memories),
            "direct_queries": sum(query["label"] == "direct" for query in queries),
            "associative_queries": sum(query["label"] == "associative" for query in queries),
            "warmup_events": len(warmup),
        },
        "sha256": {name: hashlib.sha256(payload.encode()).hexdigest() for name, payload in sorted(payloads.items())},
    }
    _write_text_exact(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_text_exact(path: Path, payload: str) -> None:
    written = path.write_text(payload, encoding="utf-8")
    if written != len(payload):
        raise OSError(f"incomplete text write: {path}")


def _select_smoke_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(memories) != FULL_MEMORY_COUNT:
        raise RuntimeError(f"smoke selection requires {FULL_MEMORY_COUNT} full memories")
    selected_ids = {
        *(memory["memory_id"] for memory in memories[:20]),
        *(memory["memory_id"] for memory in memories[100:105]),
    }
    smoke_filler_quota = {"episodic": 2, "contextual": 2, "factual": 9, "procedural": 12}
    for memory in memories[125:]:
        memory_type = memory["memory_type"]
        if smoke_filler_quota[memory_type] > 0:
            selected_ids.add(memory["memory_id"])
            smoke_filler_quota[memory_type] -= 1
        if not any(smoke_filler_quota.values()):
            break
    if any(smoke_filler_quota.values()):
        raise RuntimeError(f"could not balance smoke profile: {smoke_filler_quota}")
    selected = [memory for memory in memories if memory["memory_id"] in selected_ids]
    if len(selected) != SMOKE_MEMORY_COUNT:
        raise RuntimeError(f"smoke profile requires {SMOKE_MEMORY_COUNT} memories")
    return selected


def _select_smoke_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_query_ids = {
        "Q-001",
        "Q-002",
        "Q-003",
        "Q-004",
        "Q-005",
        "Q-026",
        "Q-027",
        "Q-028",
        "Q-029",
        "Q-030",
    }
    selected = [query for query in queries if query["query_id"] in selected_query_ids]
    expected = SMOKE_DIRECT_QUERY_COUNT + SMOKE_ASSOCIATIVE_QUERY_COUNT
    if len(selected) != expected:
        raise RuntimeError(f"smoke profile requires {expected} queries")
    return selected


def main() -> None:
    memories, queries, schedule, warmup, chains = build_full()
    _write_profile(ROOT / "full", memories, queries, schedule, warmup, chains)
    smoke_memories = _select_smoke_memories(memories)
    smoke_queries = _select_smoke_queries(queries)
    smoke_schedule = _build_schedule(smoke_memories)
    expected = (SMOKE_MEMORY_COUNT, SMOKE_DIRECT_QUERY_COUNT, SMOKE_ASSOCIATIVE_QUERY_COUNT)
    _validate_generated(smoke_memories, smoke_queries, smoke_schedule, warmup, expected)
    _write_profile(ROOT / "smoke", smoke_memories, smoke_queries, smoke_schedule, warmup)


if __name__ == "__main__":
    main()
