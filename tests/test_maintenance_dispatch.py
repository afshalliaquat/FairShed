from datetime import datetime

from src.maintenance_dispatch import MaintenanceDispatchService


def test_planned_outage_is_not_created_as_unexpected_complaint():
    service = MaintenanceDispatchService()
    result = service.submit_complaint(
        area_id="F03_COMM_A",
        hour=18,
        scheduled_off=True,
        now=datetime(2026, 11, 21, 18, 0),
    )

    assert result.accepted is False
    assert result.status == "planned_outage"
    assert result.assigned_team_id is None
    assert len(service.complaints) == 1


def test_admin_assignment_uses_ucs_and_marks_team_busy():
    service = MaintenanceDispatchService()
    submitted = service.submit_complaint(
        area_id="F03_COMM_A",
        hour=12,
        scheduled_off=False,
        now=datetime(2026, 11, 21, 12, 0),
    )

    assert submitted.status == "pending_admin"
    assert submitted.accepted is True
    assert submitted.assigned_team_id is None
    assert service.status()["complaints"][0]["status"] == "pending_admin"

    recommendation = service.get_recommended_team("F03_COMM_A")
    assert recommendation == ("TEAM_B", 5)

    assigned = service.assign_complaint(submitted.complaint_id, "TEAM_B")
    assert assigned["status"] == "dispatched"
    assert assigned["team_id"] == "TEAM_B"
    assert assigned["travel_minutes"] == 5
    assert service.teams["TEAM_B"].status == "busy"


def test_team_completion_makes_team_available_again():
    service = MaintenanceDispatchService()
    submitted = service.submit_complaint(
        area_id="F03_COMM_A",
        hour=12,
        scheduled_off=False,
    )
    assigned = service.assign_complaint(submitted.complaint_id, "TEAM_B")
    assert assigned is not None

    assert service.complete_team("TEAM_B") is True
    assert service.teams["TEAM_B"].status == "available"
    assert service.teams["TEAM_B"].assigned_area_id is None


def test_dispatch_options_compare_ucs_and_astar_for_all_available_teams():
    service = MaintenanceDispatchService()
    result = service.get_team_route_options("F03_COMM_A")

    assert result["recommended_team_id"] == "TEAM_B"
    assert result["ucs_recommended_team_id"] == "TEAM_B"
    assert result["astar_recommended_team_id"] == "TEAM_B"
    assert len(result["options"]) == 3
    assert all(
        team["ucs"]["travel_minutes"] == team["astar"]["travel_minutes"]
        for team in result["options"]
    )
