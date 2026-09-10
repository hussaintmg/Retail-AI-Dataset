# Retail AI Dataset Foundation Design

## Goal
Build a scalable, training-ready Retail AI dataset repository that supports generic retail object detection, multi-object tracking, customer-product interactions, checkout, barcode/POS reconciliation, occlusion-aware trolley state, and loss-prevention research without tying the detector to store-specific SKUs.

## Core Design Principle
The dataset must separate **generic visual detection** from **store-specific SKU recognition**.

The detector learns generic classes such as `person`, `product`, `trolley`, and `checkout_counter`. Store-specific product identity is handled later by barcode reading, visual embeddings, shelf context, and evidence fusion. This avoids retraining the detector every time a business adds a new SKU.

## Dataset Families

### 1. Detection
Purpose: generic object detection on individual frames.

Frozen class IDs for V0.1:

| ID | Class |
|---:|---|
| 0 | person |
| 1 | hand |
| 2 | product |
| 3 | shelf |
| 4 | trolley |
| 5 | basket |
| 6 | shopping_bag |
| 7 | checkout_counter |
| 8 | cashier |
| 9 | barcode_scanner |
| 10 | pos_terminal |

These IDs must never be reordered after first release. New classes may only be appended.

### 2. Tracking
Purpose: keep stable identities across frames.

Tracked entity types:
- person
- product
- trolley
- basket

Required tracking fields:
- `sequence_id`
- `camera_id`
- `frame_index`
- `timestamp_ms`
- `track_id`
- `class_id`
- `bbox_xywh`
- `visibility`
- `occlusion_level`

### 3. Interactions
Purpose: model customer-product actions over time.

Frozen V0.1 event labels:
- `pick_from_shelf`
- `return_to_shelf`
- `put_in_trolley`
- `remove_from_trolley`
- `put_in_basket`
- `remove_from_basket`
- `handover_product`
- `carry_product`
- `place_on_counter`
- `checkout_scan`
- `bag_product`
- `possible_concealment`
- `no_interaction`

`possible_concealment` is a risk-event label only. It must never be interpreted as proof of theft.

### 4. Checkout
Purpose: model hybrid AI checkout and traditional POS flows.

Checkout states:
- `shopping`
- `approaching_checkout`
- `on_counter`
- `scan_pending`
- `scanned`
- `bagged`
- `payment_pending`
- `paid`
- `exit_verified`
- `needs_review`

The dataset must support AI-expected cart items and cashier barcode scans without double-counting a sale.

### 5. Barcode
Purpose: train/test barcode visibility and decoding robustness.

Required scenarios:
- fully visible
- partial
- rotated
- upside-down
- motion blurred
- curved package
- hand occlusion
- glare
- low light
- not visible

### 6. Loss Prevention
Purpose: detect risk events and reduce false positives.

Positive/risk scenarios:
- possible concealment in bag
- possible concealment in pocket
- checkout bypass
- unpaid exit candidate

Mandatory negative scenarios:
- phone placed in pocket
- wallet placed in pocket
- personal bottle handled
- employee shelf refill
- customer inspects and returns product
- customer carries product without concealment
- child handles product
- customer-to-customer handoff
- cashier scans and repositions product

### 7. Product Recognition Reference Data
Purpose: support visual embedding search and product enrollment.

This is not a YOLO class dataset.

Per SKU store:
- SKU/product ID
- GTIN/barcode
- name
- brand
- category
- size/variant
- reference images
- optional shelf locations
- optional OCR text

Recommended views:
- front
- back
- left
- right
- top
- 45-degree views
- hand-held
- shelf view
- partial occlusion
- alternate lighting

## Occlusion and Persistent Cart State
A product must remain logically inside a trolley/basket even when it becomes fully hidden.

Example state flow:

`VISIBLE -> PICKED -> IN_HAND -> ENTERED_TROLLEY -> OCCLUDED_INSIDE_TROLLEY -> STILL_IN_TROLLEY -> REMOVED | CHECKED_OUT | RETURNED`

Required temporal fields:
- `product_track_id`
- `person_track_id`
- `container_track_id`
- `camera_id`
- `frame_index`
- `visibility`
- `occlusion_level`
- `event`
- `state_before`
- `state_after`
- `location`
- `checkout_id`
- `transaction_id`

A track disappearing from pixels is not sufficient to remove it from cart state. Only an explicit state-changing event may do so.

## Repository Layout

```text
Retail-AI-Dataset/
├── README.md
├── configs/
│   ├── data.yaml
│   ├── classes.yaml
│   └── events.yaml
├── schemas/
│   ├── detection.schema.json
│   ├── tracking.schema.json
│   ├── interaction.schema.json
│   ├── checkout.schema.json
│   └── product_catalog.schema.json
├── datasets/
│   ├── detection/
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── tracking/
│   ├── interactions/
│   ├── checkout/
│   ├── barcode/
│   ├── loss_prevention/
│   ├── products/
│   └── synthetic/
├── manifests/
│   ├── batches/
│   └── versions/
├── scripts/
│   ├── validate_dataset.py
│   ├── validate_yolo_labels.py
│   ├── validate_tracking.py
│   ├── build_manifest.py
│   └── split_sequences.py
├── tests/
└── docs/
```

Empty directories will be represented with `.gitkeep` files because Git does not track empty folders.

## YOLO Format
Detection labels use standard normalized YOLO bounding-box format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Rules:
- coordinates must be in `[0,1]`
- width/height must be greater than `0`
- class IDs must exist in `classes.yaml`
- one annotation row per object instance
- every image must have a matching label file; negative images use an empty `.txt`

## Tracking Format
Tracking data will use a project-owned JSONL format as the source of truth, with export adapters later for MOTChallenge-style tooling.

One JSON object per frame/entity observation.

Example:

```json
{"sequence_id":"seq_000001","camera_id":"cam_01","frame_index":120,"track_id":"product_000044","class_id":2,"bbox_xywh":[0.42,0.31,0.08,0.12],"visibility":"partial","occlusion_level":0.45}
```

## Interaction Format
Interaction annotations are temporal events, not per-frame detector classes.

Example:

```json
{"event_id":"evt_000001","sequence_id":"seq_000001","event":"put_in_trolley","start_frame":118,"end_frame":136,"person_track_id":"person_000012","product_track_id":"product_000044","container_track_id":"trolley_000003","state_before":"in_hand","state_after":"in_trolley","verified":true}
```

## Train / Validation / Test Split
Default split: 70 / 15 / 15.

Hard rule: adjacent frames from the same source video must not be randomly split across train/val/test.

Split by source sequence, person/session, camera setup, day, or synthetic seed to avoid leakage.

## Dataset QA
Every batch must pass automated checks before it is marked accepted.

Validation includes:
- duplicate file detection
- missing image/label pairs
- invalid class IDs
- invalid normalized boxes
- zero-area boxes
- corrupt/empty image references
- sequence frame-order checks
- duplicate track IDs in one frame for the same entity
- impossible state transitions
- unknown event labels
- train/val/test source leakage
- manifest count consistency

Failed samples go into a quarantine/review report and are not counted as accepted dataset items.

## Batch Strategy
Do not jump directly to millions of frames.

Release ladder:
- V0: 5k-20k images/frames to validate architecture and labels
- V1: 50k-100k for detection and tracking robustness
- V2: 250k-500k for interactions and occlusions
- V3: 1M+ using synthetic retail generation plus real data
- V4: multi-million, multi-store, multi-camera diversity

Each batch manifest must record:
- batch ID
- source type (`real`, `synthetic`, `augmented`)
- accepted image/frame count
- rejected count
- annotation count
- class distribution
- event distribution
- sequences
- camera IDs
- split assignment
- generator/version information
- checksum metadata
- QA result

## Synthetic Data Strategy
Synthetic data will eventually generate scalable store diversity while keeping exact ground truth.

Scenario generator must vary:
- store layout
- aisle count
- shelf layout
- product placement
- trolley/basket type
- camera height/angle/lens
- lighting
- compression
- motion blur
- crowd density
- clothes/appearance
- package orientation
- occlusion

Synthetic labels may include exact object IDs, bounding boxes, segmentation masks if available, product IDs, person IDs, cart IDs, 3D locations, visibility, actions, and timestamps.

## Real/Synthetic Balance
Synthetic data is not a replacement for real footage.

Real data is required for:
- CCTV compression artifacts
- difficult glare
- real occlusion patterns
- human motion
- store-specific camera placement
- real package appearance
- production validation

Synthetic data is used to scale coverage after label architecture is proven.

## Product Identity Rule
Do not add Coke, Pepsi, Lays, milk brands, etc. as generic detector classes.

Product identity should be resolved by evidence fusion:
1. barcode if readable
2. visual embedding similarity
3. OCR/logo evidence
4. shelf location/context
5. track history
6. multi-camera evidence
7. low confidence -> unknown/review

## Unknown Handling
The system must allow `UNKNOWN_PRODUCT` rather than forcing a wrong SKU match.

Human review can:
- assign an existing product
- create a new product
- mark the crop unusable

Verified corrections may later be promoted into the product reference gallery or training dataset.

## Versioning Rules
- Class IDs are append-only.
- Event labels are append-only once a release is published.
- Every accepted batch gets an immutable manifest.
- Dataset releases reference immutable batch IDs.
- Corrections produce a new dataset version rather than silently mutating a published release.
- Training runs must record the exact dataset version and config used.

## Initial Implementation Scope
The first implementation milestone will create:
- repository folders
- `classes.yaml`
- `events.yaml`
- YOLO `data.yaml`
- JSON schemas
- validation scripts
- tests for validators
- `.gitkeep` placeholders
- Batch 001 manifest template
- initial README and contributor/annotation rules

The first milestone does not claim to create thousands of unique real retail photos from nothing. It creates the production-safe dataset factory and label system so real, synthetic, and generated samples can be added in large validated batches without corrupting the taxonomy.
