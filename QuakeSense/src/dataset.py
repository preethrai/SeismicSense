import random
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from .config import (NETWORK, STATION_CANDIDATES, CHANNEL, START_TIME, END_TIME,
                     MIN_MAGNITUDE, WINDOW_SECONDS, TARGET_SAMPLE_RATE,
                     WARMUP_SECONDS, MAX_POSITIVES, NEG_TO_POS_RATIO, OUT_PREFIX)
from .preprocessing import preprocess

event_client = Client("USGS")
wave_client = Client("EARTHSCOPE")


def fetch_events():

    print("Fetching earthquake catalog from USGS...")

    catalog = event_client.get_events(
        starttime=START_TIME,
        endtime=END_TIME,
        minmagnitude=MIN_MAGNITUDE,
        minlatitude=35.4,
        maxlatitude=35.9,
        minlongitude=-117.8,
        maxlongitude=-117.3,
    )

    print(
        f"Found {len(catalog)} events."
    )

    return catalog
def get_window(start_time):

    """
    start_time = beginning of the 30-second evaluation window.

    Fetch:

        start_time - 25s
                ↓
        ┌──────────────┬──────────────────────────┐
        │   WARMUP     │       EVALUATION         │
        │     25s      │           30s            │
        └──────────────┴──────────────────────────┘
                       ↑
                   start_time
    """

    fetch_start = (
        start_time -
        WARMUP_SECONDS
    )

    fetch_end = (
        start_time +
        WINDOW_SECONDS
    )


    for station in STATION_CANDIDATES:

        try:

            st = wave_client.get_waveforms(
                NETWORK,
                station,
                "",
                CHANNEL,
                fetch_start,
                fetch_end
            )

        except Exception:

            continue


        if len(st) == 0:

            continue


        try:

            (
                raw_data,
                normalized_data,
                warmup,
                baseline_features,
                scale
            ) = preprocess(st[0])

        except Exception:

            continue


        return (
            raw_data,
            normalized_data,
            warmup,
            baseline_features,
            scale,
            station
        )


    return (
        None,
        None,
        None,
        None,
        None,
        None
    )
def build_positives(catalog):

    raw_samples = []

    ml_samples = []

    warmup_samples = []

    baseline_features = []

    normalization_scales = []

    meta = []


    events = list(catalog)[
        :MAX_POSITIVES
    ]


    for i, event in enumerate(events):

        origin = (
            event.preferred_origin()
            or event.origins[0]
        )

        event_time = origin.time


        # Evaluation begins 5 seconds BEFORE
        # the earthquake origin.

        window_start = (
            event_time - 5
        )


        result = get_window(
            window_start
        )


        (
            raw_data,
            normalized_data,
            warmup,
            baseline,
            scale,
            station
        ) = result


        if raw_data is None:

            print(
                f"  skipped event {i} "
                f"({event_time}): "
                f"no station had data"
            )

            continue


        raw_samples.append(
            raw_data
        )

        ml_samples.append(
            normalized_data
        )

        warmup_samples.append(
            warmup
        )

        baseline_features.append(
            baseline
        )

        normalization_scales.append(
            scale
        )


        magnitude = None

        if event.preferred_magnitude():

            magnitude = (
                event.preferred_magnitude().mag
            )


        meta.append({

            "label": 1,

            "event_id": str(
                event.resource_id
            ),

            "station": station,

            "start_time": str(
                window_start
            ),

            "magnitude": magnitude,

        })


        print(
            f"  positive "
            f"{len(raw_samples)}/"
            f"{len(events)} "
            f"(station={station})"
        )


    return (
        raw_samples,
        ml_samples,
        warmup_samples,
        baseline_features,
        normalization_scales,
        meta
    )


# ---------------------------------------------------------------------------
def build_negatives(
    catalog,
    n_needed
):

    event_times = [

        (
            e.preferred_origin()
            or e.origins[0]
        ).time

        for e in catalog

    ]


    raw_samples = []

    ml_samples = []

    warmup_samples = []

    baseline_features = []

    normalization_scales = []

    meta = []


    total_seconds = int(
        END_TIME - START_TIME
    )

    attempts = 0


    while (
        len(raw_samples) < n_needed
        and attempts < n_needed * 10
    ):

        attempts += 1


        random_offset = random.randint(
            0,
            total_seconds -
            WINDOW_SECONDS
        )


        candidate_time = (
            START_TIME +
            random_offset
        )


        # Keep negative samples away from
        # catalogued earthquakes.

        if any(
            abs(candidate_time - et) < 60
            for et in event_times
        ):

            continue


        result = get_window(
            candidate_time
        )


        (
            raw_data,
            normalized_data,
            warmup,
            baseline,
            scale,
            station
        ) = result


        if raw_data is None:

            continue


        raw_samples.append(
            raw_data
        )

        ml_samples.append(
            normalized_data
        )

        warmup_samples.append(
            warmup
        )

        baseline_features.append(
            baseline
        )

        normalization_scales.append(
            scale
        )


        meta.append({

            "label": 0,

            "event_id": None,

            "station": station,

            "start_time": str(
                candidate_time
            ),

            "magnitude": None,

        })


        print(
            f"  negative "
            f"{len(raw_samples)}/"
            f"{n_needed} "
            f"(station={station})"
        )


    return (
        raw_samples,
        ml_samples,
        warmup_samples,
        baseline_features,
        normalization_scales,
        meta
    )


