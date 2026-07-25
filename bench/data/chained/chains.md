# Synthetic Associative Query Chains

All organizations, systems, incidents, and values in this dataset are fictional.

Each chain carries two questions. The **holdout** query asks about the
symptom and expects the chain's target; the **chained warm-up** query asks
about the mechanism and expects the first bridge. The `chained` profile
replays the warm-up questions so co-retrieval and feedback act on the same
memories the holdout later needs, without ever surfacing the answer.

## Q-026

**Holdout query:** Why do refrigerated shipments lose temperature readings after leaving the Alder depot?

**Chained warm-up query:** Which radio modules did the Alder depot gateways receive in the spring retrofit? (expects `mem-0002`)

**Chain:**

- Anchor `mem-0001`: Temperature readings from insulated crates disappear for several minutes whenever trucks leave the Alder depot.
- Intermediate `mem-0002`: Alder depot gateways forward crate telemetry through Zephyr radio modules installed during the spring retrofit.
- Intermediate `mem-0003`: Zephyr firmware 4.2 suspends radio polling when its eco-sleep flag detects external power loss, even though the gateway battery remains healthy.
- **Target** `mem-0004`: The fleet profile now keeps radio polling active during battery operation; packet completeness returned to 99.8 percent on the next twelve routes.

*Note: The apparent sensor gap leads through the depot gateway and its radio power policy.*

---

## Q-027

**Holdout query:** What is causing incorrect product codes during picking in zone C?

**Chained warm-up query:** Which layout profile configures the label printers used in zone C? (expects `mem-0006`)

**Chain:**

- Anchor `mem-0005`: Pickers in zone C scan the correct shelf but receive a neighboring product code roughly once every thirty orders.
- Intermediate `mem-0006`: Zone C uses compact label printers configured by the Quartz layout profile rather than the standard warehouse profile.
- Intermediate `mem-0007`: The Quartz profile was exported at 203 DPI while the replacement printers render at 300 DPI, shifting the barcode crop box by four millimeters.
- **Target** `mem-0008`: A 300-DPI layout was deployed and the crop coordinates were regenerated; the following 4,800 picks had no adjacent-label substitutions.

*Note: The scan errors trace from the zone-specific printer profile to a resolution mismatch.*

---

## Q-028

**Holdout query:** Why are rescheduled afternoon visits appearing twice on clinic calendars?

**Chained warm-up query:** Which scheduling worker handles portal changes before they reach practitioner calendars? (expects `mem-0010`)

**Chain:**

- Anchor `mem-0009`: Some afternoon appointments appear twice on practitioner calendars after patients reschedule through the mobile portal.
- Intermediate `mem-0010`: Portal changes pass through the Saffron scheduling worker before they are written to each practitioner's calendar.
- Intermediate `mem-0011`: Saffron compares local appointment strings before converting them to UTC, so daylight-saving offsets make the old and new slots look distinct.
- **Target** `mem-0012`: The worker now normalizes both timestamps before comparison and a cleanup removed the duplicate slots without changing valid bookings.

*Note: The duplicate booking symptom is connected to the scheduling worker's timestamp comparison order.*

---

## Q-029

**Holdout query:** Why do the east orchard rows start watering progressively later?

**Chained warm-up query:** Which timing board does the east orchard field controller run? (expects `mem-0014`)

**Chain:**

- Anchor `mem-0013`: Irrigation valves in the east rows begin their morning cycle almost twenty minutes later each week.
- Intermediate `mem-0014`: The east controller is the only field unit using the Solace solar timing board introduced in last year's expansion.
- Intermediate `mem-0015`: Solace board revision B loses oscillator compensation whenever its charge controller enters low-light mode overnight.
- **Target** `mem-0016`: Installing the compensated timing package stopped the accumulated drift; valve starts stayed within thirty seconds for six weeks.

*Note: The growing schedule delay leads through the unique solar controller to its uncompensated clock.*

---

## Q-030

**Holdout query:** Why are westbound cargo updates stale despite successful station scans?

**Chained warm-up query:** Which broker group consumes the westbound station updates? (expects `mem-0018`)

**Chain:**

- Anchor `mem-0017`: Cargo status for westbound trains remains unchanged for hours even though station scans are completing normally.
- Intermediate `mem-0018`: Westbound updates are consumed by the Marigold broker group shared with the overnight reconciliation job.
- Intermediate `mem-0019`: The reconciliation deployment accidentally reused the live consumer-group name, allowing it to claim and acknowledge status partitions.
- **Target** `mem-0020`: Giving reconciliation a separate consumer group restored live status flow and replayed the unprocessed station events.

*Note: The stale status connects through shared message consumption to a colliding consumer-group identity.*

---

## Q-031

**Holdout query:** Why do audio guides fail only in the museum's west gallery?

**Chained warm-up query:** Which proximity beacons were installed in the west gallery after the construction work? (expects `mem-0022`)

**Chain:**

- Anchor `mem-0021`: Audio guides stop recognizing exhibits in the new west gallery while older galleries continue to work.
- Intermediate `mem-0022`: The west gallery received a replacement set of Juniper proximity beacons after construction dust damaged the originals.
- Intermediate `mem-0023`: The replacement beacons rotate identifiers from a new namespace that is absent from the guide application's offline exhibit manifest.
- **Target** `mem-0024`: Publishing the new beacon namespace in manifest revision 18 restored exhibit recognition without an application release.

*Note: The gallery-specific failure leads through replacement beacons to an outdated offline identifier manifest.*

---

## Q-032

**Holdout query:** Why do newly issued hotel key cards open rooms but not elevators?

**Chained warm-up query:** Which access server do the hotel elevator readers validate credentials against? (expects `mem-0026`)

**Chain:**

- Anchor `mem-0025`: Guest key cards issued after noon fail at elevator readers but still open room doors.
- Intermediate `mem-0026`: Elevator readers validate credentials through the Tidepool access server, while room locks validate cards locally.
- Intermediate `mem-0027`: Tidepool's midday certificate rollover installed a new leaf certificate without the intermediate authority expected by elevator firmware.
- **Target** `mem-0028`: The complete trust chain was added to the access-server bundle and newly issued cards began working at every elevator reader.

*Note: The split behavior leads through server-based elevator validation to an incomplete certificate chain.*

---

## Q-033

**Holdout query:** Why are morning loaves dense when the recipe and mixing process are unchanged?

**Chained warm-up query:** Which sensor cartridge reports humidity in proofing cabinet seven? (expects `mem-0030`)

**Chain:**

- Anchor `mem-0029`: Morning bread batches are unusually dense even though mixing times and ingredient weights match the recipe.
- Intermediate `mem-0030`: Proofing cabinet seven reports humidity through a recently replaced Nimbus sensor cartridge.
- Intermediate `mem-0031`: The replacement cartridge emits a fractional value while the cabinet controller interprets its input as a percentage, reducing steam output by a factor of one hundred.
- **Target** `mem-0032`: A conversion factor was added to the cabinet profile; loaf volume returned to its normal range in the next production run.

*Note: The texture problem leads through proofing conditions to a humidity-unit mismatch.*

---

## Q-034

**Holdout query:** Why do dock-three boarding scanners disconnect just before departure?

**Chained warm-up query:** Which network template was dock three moved onto during its renovation? (expects `mem-0034`)

**Chain:**

- Anchor `mem-0033`: Boarding scanners at dock three lose connectivity during the final ten minutes before departure.
- Intermediate `mem-0034`: Dock three was moved onto the Coral network template when its waiting area was renovated.
- Intermediate `mem-0035`: The Coral template assigns handheld devices to a guest VLAN whose address lease is shorter than the boarding window.
- **Target** `mem-0036`: Applying the transit-device VLAN profile eliminated lease expiration during boarding and stabilized all six scanners.

*Note: The time-specific disconnect leads through the renovated dock network to an unsuitable lease policy.*

---

## Q-035

**Holdout query:** Why is the recycling line misclassifying transparent containers?

**Chained warm-up query:** Which camera handles transparent-item classification on the sorting line? (expects `mem-0038`)

**Chain:**

- Anchor `mem-0037`: The optical sorter recently started sending transparent containers into the mixed-waste lane.
- Intermediate `mem-0038`: Transparent-item classification uses camera three, which received a replacement lens during scheduled cleaning.
- Intermediate `mem-0039`: The lens installation reset camera three to automatic exposure, washing out the edge contrast used by the material classifier.
- **Target** `mem-0040`: Restoring the fixed exposure profile raised transparent-container classification to 98.6 percent during the validation shift.

*Note: The material error connects through a serviced camera to an exposure-profile reset.*

---

## Q-036

**Holdout query:** Why are science-building room schedules blank on Monday mornings?

**Chained warm-up query:** Which campus service provides the weekly schedule manifest to the displays? (expects `mem-0042`)

**Chain:**

- Anchor `mem-0041`: Room displays across the science building show a blank schedule every Monday morning.
- Intermediate `mem-0042`: The displays load a weekly schedule manifest from the Rowan campus content service.
- Intermediate `mem-0043`: Sunday publishing reuses the same manifest filename, and the building cache retains the previous week's expired response until noon Monday.
- **Target** `mem-0044`: Versioned manifest filenames and a publish-time purge made new schedules visible immediately on all room displays.

*Note: The weekly display failure leads through manifest delivery to a stale building cache.*

---

## Q-037

**Holdout query:** Why are some successfully scanned laboratory samples missing their intake records?

**Chained warm-up query:** What identifier format do the partner clinics use on their sample tubes? (expects `mem-0046`)

**Chain:**

- Anchor `mem-0045`: Several sample results cannot be matched to intake records even though technicians scan every tube successfully.
- Intermediate `mem-0046`: The affected tubes come from partner clinics using six-character identifiers that begin with zero.
- Intermediate `mem-0047`: The Fern import worker converts scanned identifiers to integers before lookup, permanently removing leading zeros.
- **Target** `mem-0048`: Identifier handling was changed to validated strings and the unmatched results were reconciled from the audit queue.

*Note: The matching failure leads through a partner identifier format to destructive numeric conversion.*

---

## Q-038

**Holdout query:** Why are south-site drone maps vertically offset from surveyed control points?

**Chained warm-up query:** Which conversion profile does the south project use to import positioning data? (expects `mem-0050`)

**Chain:**

- Anchor `mem-0049`: Drone elevation maps for the south construction site are consistently about thirty meters above ground-control measurements.
- Intermediate `mem-0050`: The south project imports positioning data through the Aster terrain conversion profile.
- Intermediate `mem-0051`: Aster uses ellipsoidal heights while the site's control points use the regional geoid model, creating a nearly constant vertical offset.
- **Target** `mem-0052`: Switching the conversion profile to the regional geoid aligned drone surfaces with every surveyed control point.

*Note: The constant elevation error leads through the terrain profile to incompatible vertical datums.*

---

## Q-039

**Holdout query:** Why do theater subtitles gradually fall behind even though every act starts synchronized?

**Chained warm-up query:** Which audio feed drives subtitle timing for the touring production? (expects `mem-0054`)

**Chain:**

- Anchor `mem-0053`: Projected subtitles drift behind dialogue as a performance progresses, but they begin each act in sync.
- Intermediate `mem-0054`: Subtitle timing is driven by the Indigo audio feed installed with the touring production.
- Intermediate `mem-0055`: Indigo declares a 48-kHz stream while the interface actually delivers 47.952 kHz, causing cumulative timing drift.
- **Target** `mem-0056`: Resampling the feed to the declared rate kept subtitles aligned through the full three-hour dress rehearsal.

*Note: The cumulative drift connects through the touring audio feed to a sample-rate discrepancy.*

---

## Q-040

**Holdout query:** Why does food-bank inventory become negative during busy distribution periods?

**Chained warm-up query:** Which retry queue carries stock decrements from the mobile checkout tablets? (expects `mem-0058`)

**Chain:**

- Anchor `mem-0057`: Inventory for several staple items becomes negative during the busiest distribution window.
- Intermediate `mem-0058`: Mobile checkout tablets send stock decrements through the Clover retry queue when wireless service is congested.
- Intermediate `mem-0059`: Clover retries timed-out requests without preserving an operation key, so completed decrements can be applied a second time.
- **Target** `mem-0060`: Checkout now assigns an immutable operation key to each decrement; duplicate retries are ignored and stock totals remain nonnegative.

*Note: The stock anomaly leads through congested tablet retries to missing operation idempotency.*

---

## Q-041

**Holdout query:** Why does the observatory dome trigger wind protection during calm weather?

**Chained warm-up query:** Which weather adapter feeds wind data to the dome controller? (expects `mem-0062`)

**Chain:**

- Anchor `mem-0061`: The dome closes for high-wind protection on calm evenings several times each month.
- Intermediate `mem-0062`: Wind data reaches the dome controller through the Boreal weather adapter added for the new roof station.
- Intermediate `mem-0063`: Boreal labels its readings as meters per second but forwards the station's original knot values unchanged.
- **Target** `mem-0064`: Converting knots before publishing reduced false closure alerts while preserving every genuinely windy shutdown.

*Note: The false safety response leads through the weather adapter to mislabeled units.*

---

## Q-042

**Holdout query:** Why do garden kiosks forget a visitor's language selection between pages?

**Chained warm-up query:** Which web client stores the language preference on the outdoor kiosk tablets? (expects `mem-0066`)

**Chain:**

- Anchor `mem-0065`: Visitor kiosks return to the default language between pages despite a guest selecting another language.
- Intermediate `mem-0066`: Language preference is stored by the Petal web client loaded on outdoor kiosk tablets.
- Intermediate `mem-0067`: Petal writes the preference to a secure cookie, but the outdoor kiosks serve local pages without a secure transport context.
- **Target** `mem-0068`: Moving the preference to local kiosk storage preserved language selection throughout each visitor session.

*Note: The reset behavior leads through client-side preference storage to an unusable secure cookie.*

---

## Q-043

**Holdout query:** Why does the bicycle map show full stations after bicycles have been removed?

**Chained warm-up query:** Which collector reports dock occupancy to the bicycle map? (expects `mem-0070`)

**Chain:**

- Anchor `mem-0069`: The map reports popular docking stations as full even after riders remove several bicycles.
- Intermediate `mem-0070`: Dock occupancy reaches the map through the Sparrow heartbeat collector.
- Intermediate `mem-0071`: Sparrow batches fifty heartbeats, but popular stations exceed the collector's flush timeout before a batch fills during uneven traffic.
- **Target** `mem-0072`: Reducing the batch size and adding a two-second flush updated station capacity promptly throughout the evening commute.

*Note: The stale capacity display leads through heartbeat collection to an oversized batching rule.*

---

## Q-044

**Holdout query:** Why is kiln four overheating while its display appears correct?

**Chained warm-up query:** When did kiln four receive its current temperature probe? (expects `mem-0074`)

**Chain:**

- Anchor `mem-0073`: Kiln four overshoots firing targets by nearly eighty degrees while its control panel reports the requested temperature.
- Intermediate `mem-0074`: Kiln four received a new temperature probe during the last refractory-lining replacement.
- Intermediate `mem-0075`: The installed probe is type K, but the controller channel remains configured for the type J probe it replaced.
- **Target** `mem-0076`: Changing the controller channel to type K brought reference-cone readings and panel temperatures back into agreement.

*Note: The hidden temperature error leads through a replaced probe to the wrong thermocouple profile.*

---

## Q-045

**Holdout query:** Why did trail-camera battery life collapse after the winter survey update?

**Chained warm-up query:** Which motion package was installed on the trail cameras during the winter survey? (expects `mem-0078`)

**Chain:**

- Anchor `mem-0077`: Remote trail cameras now exhaust their batteries in under a week instead of lasting two months.
- Intermediate `mem-0078`: The affected cameras received the Wren motion package during the winter animal-count survey.
- Intermediate `mem-0079`: Wren treats swaying vegetation as a fresh trigger on every frame because its debounce state resets after each short recording.
- **Target** `mem-0080`: Persisting debounce state across recordings reduced false captures and restored the expected battery lifetime.

*Note: The power drain leads through the motion package to repeated vegetation triggers.*

---

## Q-046

**Holdout query:** Why do campground reservations temporarily vanish from the public availability board?

**Chained warm-up query:** Which data source does the public availability board read from? (expects `mem-0082`)

**Chain:**

- Anchor `mem-0081`: New reservations occasionally disappear from the availability board for ten to fifteen minutes.
- Intermediate `mem-0082`: The public board reads from the Driftwood reservation replica rather than the booking database.
- Intermediate `mem-0083`: A nightly campsite-pricing refresh holds one large transaction on Driftwood, delaying replication of concurrent bookings.
- **Target** `mem-0084`: Chunking the pricing refresh into small commits kept replication delay below three seconds during subsequent runs.

*Note: The delayed visibility leads through the read replica to a long-running pricing transaction.*

---

## Q-047

**Holdout query:** Why are aquarium filtration pumps rapidly cycling under stable pressure?

**Chained warm-up query:** Which pressure sensor informs the filtration pump decisions? (expects `mem-0086`)

**Chain:**

- Anchor `mem-0085`: Filtration pumps cycle on and off every few seconds even though tank pressure is stable.
- Intermediate `mem-0086`: Pump decisions use readings from the Cascade pressure sensor installed beside the main return pipe.
- Intermediate `mem-0087`: Cascade readings fluctuate by a tiny amount around the exact switch threshold, and the controller has no hysteresis band.
- **Target** `mem-0088`: Adding a pressure deadband stopped rapid cycling without changing normal filtration response.

*Note: The mechanical cycling leads through threshold readings to missing controller hysteresis.*

---

## Q-048

**Holdout query:** Why does the radio station repeat its opening songs every morning?

**Chained warm-up query:** Which scheduler handles song selection at the station? (expects `mem-0090`)

**Chain:**

- Anchor `mem-0089`: The morning program repeats the same opening songs on consecutive days despite a large approved playlist.
- Intermediate `mem-0090`: Song selection is handled by the Thistle scheduler restarted during each overnight maintenance window.
- Intermediate `mem-0091`: Thistle initializes its playlist cursor from the same daily seed whenever no saved cursor is present.
- **Target** `mem-0092`: Persisting the cursor before maintenance produced varied openings while retaining the approved rotation rules.

*Note: The repeated sequence leads through overnight scheduler restarts to an unsaved playlist cursor.*

---

## Q-049

**Holdout query:** Why are textile dye batches lighter when pigment quantities have not changed?

**Chained warm-up query:** Which probe reports the dye-vat temperature? (expects `mem-0094`)

**Chain:**

- Anchor `mem-0093`: Indigo dye batches have become visibly lighter even though pigment measurements remain unchanged.
- Intermediate `mem-0094`: Dye-vat temperature is reported by the Flax probe calibrated during the annual equipment inspection.
- Intermediate `mem-0095`: The inspection worksheet applied the ambient-water offset twice when programming the Flax probe.
- **Target** `mem-0096`: Removing the duplicate offset restored the target shade across three validation batches.

*Note: The shade change leads through vat conditions to a duplicated probe calibration offset.*

---

## Q-050

**Holdout query:** Why do relief-supply pallet counts double after offline scanners reconnect?

**Chained warm-up query:** Which field queue replays offline scans when a distribution center reconnects? (expects `mem-0098`)

**Chain:**

- Anchor `mem-0097`: The emergency-supply dashboard shows duplicate pallet counts after scanners reconnect from offline mode.
- Intermediate `mem-0098`: Offline scans are replayed through the Acorn field queue when a distribution center regains service.
- Intermediate `mem-0099`: Acorn assigns a new event identifier during replay instead of preserving the scanner's original identifier.
- **Target** `mem-0100`: Preserving original event identifiers allowed the inventory service to discard replayed duplicates automatically.

*Note: The duplicate inventory leads through offline replay to regenerated event identities.*

---
