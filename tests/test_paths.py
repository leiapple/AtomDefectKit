from atomdefectkit.utils.paths import WorkingDirectoryMixin, ensure_working_dir, working_path


def test_ensure_working_dir_creates_nested_path(tmp_path):
    output_dir = ensure_working_dir(tmp_path, "model", "run")

    assert (tmp_path / "model" / "run").is_dir()
    assert output_dir == str(tmp_path / "model" / "run")


def test_working_path_joins_without_creating(tmp_path):
    output_path = working_path(tmp_path, "result.txt")

    assert output_path == str(tmp_path / "result.txt")
    assert not (tmp_path / "result.txt").exists()


def test_working_directory_mixin_creates_and_joins_paths(tmp_path):
    workflow = WorkingDirectoryMixin()

    workflow.init_working_dir(tmp_path, "workflow")

    assert (tmp_path / "workflow").is_dir()
    assert workflow.path("log.txt") == str(tmp_path / "workflow" / "log.txt")
