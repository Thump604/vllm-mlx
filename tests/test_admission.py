# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from vllm_mlx.admission import AdmissionCapacityError, AdmissionController


def test_request_limit_rejects_without_changing_reservations():
    admission = AdmissionController(max_requests=1)
    admission.reserve("first", 3)

    with pytest.raises(AdmissionCapacityError) as excinfo:
        admission.reserve("second", 2)

    assert excinfo.value.resource == "request"
    assert admission.snapshot().num_requests == 1
    assert admission.snapshot().num_prompt_tokens == 3


def test_prompt_token_limit_accepts_boundary_and_rejects_next_token():
    admission = AdmissionController(max_prompt_tokens=5)
    admission.reserve("first", 3)
    admission.reserve("second", 2)

    with pytest.raises(AdmissionCapacityError) as excinfo:
        admission.reserve("third", 1)

    assert excinfo.value.resource == "prompt_token"
    assert admission.snapshot().num_requests == 2
    assert admission.snapshot().num_prompt_tokens == 5


def test_release_is_idempotent_and_restores_capacity():
    admission = AdmissionController(max_requests=1, max_prompt_tokens=3)
    admission.reserve("first", 3)

    assert admission.release("first") is True
    assert admission.release("first") is False

    admission.reserve("second", 3)
    assert admission.snapshot().num_requests == 1
    assert admission.snapshot().num_prompt_tokens == 3


def test_unlimited_defaults_accept_multiple_reservations():
    admission = AdmissionController()

    for index in range(100):
        admission.reserve(str(index), index)

    assert admission.snapshot().num_requests == 100
    assert admission.snapshot().num_prompt_tokens == sum(range(100))


def test_duplicate_request_id_does_not_change_accounting():
    admission = AdmissionController()
    admission.reserve("same", 2)

    with pytest.raises(ValueError, match="already has a reservation"):
        admission.reserve("same", 3)

    assert admission.snapshot().num_requests == 1
    assert admission.snapshot().num_prompt_tokens == 2


def test_zero_token_release_is_accounted_as_a_real_reservation():
    admission = AdmissionController(max_requests=1, max_prompt_tokens=1)
    admission.reserve("empty", 0)

    assert admission.release("empty") is True
    assert admission.snapshot().num_requests == 0
    assert admission.snapshot().num_prompt_tokens == 0


def test_clear_releases_active_reservations():
    admission = AdmissionController(max_requests=2, max_prompt_tokens=5)
    admission.reserve("first", 2)
    admission.reserve("second", 3)

    admission.clear()

    assert admission.snapshot().num_requests == 0
    assert admission.snapshot().num_prompt_tokens == 0
    admission.reserve("replacement", 5)
    assert admission.snapshot().num_requests == 1
    assert admission.snapshot().num_prompt_tokens == 5


def test_capacity_error_exposes_machine_readable_details():
    admission = AdmissionController(max_prompt_tokens=4)
    admission.reserve("first", 3)

    with pytest.raises(AdmissionCapacityError) as excinfo:
        admission.reserve("second", 2)

    error = excinfo.value
    assert error.code == "scheduler_capacity_exceeded"
    assert error.resource == "prompt_token"
    assert error.limit == 4
    assert error.current == 3
    assert error.requested == 2


def test_concurrent_reservations_observe_one_request_boundary():
    admission = AdmissionController(max_requests=1)
    barrier = threading.Barrier(2)
    outcomes = []

    def reserve(request_id: str) -> None:
        barrier.wait()
        try:
            admission.reserve(request_id, 0)
        except AdmissionCapacityError as error:
            outcomes.append((request_id, error.current, error.requested))
        else:
            outcomes.append((request_id, "accepted"))

    threads = [
        threading.Thread(target=reserve, args=(request_id,))
        for request_id in ("first", "second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcome[1] for outcome in outcomes if outcome[1] == "accepted") == [
        "accepted"
    ]
    rejected = [outcome for outcome in outcomes if outcome[1] != "accepted"]
    assert len(rejected) == 1
    assert rejected[0][1:] == (1, 1)
    assert admission.snapshot().num_requests == 1


@pytest.mark.parametrize("value", [0, -1])
def test_limits_must_be_positive_or_none(value):
    with pytest.raises(ValueError):
        AdmissionController(max_requests=value)

    with pytest.raises(ValueError):
        AdmissionController(max_prompt_tokens=value)
