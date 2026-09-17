'''MAS3K 多 GPU 启动器的分片与命令测试。'''

from pathlib import Path

import run_mas3k_multi_gpu as launcher


def test_parse_gpu_ids_supports_spaces_and_commas():
    '''GPU 参数同时支持空格和逗号写法。'''

    assert launcher.parse_gpu_ids(["2,3", "4"]) == ["2", "3", "4"]


def test_split_images_is_balanced_and_complete():
    '''分片均衡、不重复且覆盖全部输入。'''

    images = [Path(f"image_{index}.jpg") for index in range(7)]
    shards = launcher.split_images(images, 3)
    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert sorted(path for shard in shards for path in shard) == images


def test_worker_command_overrides_gpu_and_summary(tmp_path):
    '''worker 使用独立 GPU 与摘要，同时共享结果和最终 mask 目录。'''

    args = launcher.build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    command = launcher.build_worker_command(
        args,
        "2",
        tmp_path / "shard.txt",
        tmp_path / "summary.json",
        tmp_path / "masks",
    )
    assert command[command.index("--gpu") + 1] == "2"
    assert command[command.index("--summary-path") + 1] == str(
        tmp_path / "summary.json"
    )
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "output")
