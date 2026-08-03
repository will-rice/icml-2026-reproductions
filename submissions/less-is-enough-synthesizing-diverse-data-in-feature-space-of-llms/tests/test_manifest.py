from fac_evidence.manifest import build_manifest


def test_upstream_manifest_records_inaccessible_dataset_api():
    manifest = build_manifest()

    dataset = manifest["artifacts"]["hf_dataset_api"]

    assert dataset["status_code"] == 401
    assert dataset["access"] == "unavailable"
    assert manifest["artifacts"]["github"]["revision"] == "d3622dec55123c0eff4c079db9e1a59403f08d1b"
