'''验证类别提取、可选规范化和旧配置兼容性。'''

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sam3_agent.batch.prompts import extract_prompt
from sam3_agent.config.schema import PromptConfig


@pytest.fixture
def mas3k_prompt_config():
    '''使用项目实际 MAS3K 配置，避免测试规则与入口配置脱节。'''

    config_path = Path(__file__).parents[1] / "agent_config.yaml"
    prompt = yaml.safe_load(config_path.read_text())["prompt"]
    return PromptConfig(**prompt)


@pytest.mark.parametrize("scene", ["Cam", "Com"])
@pytest.mark.parametrize(
    "category,expected",
    [
        ("Arthropod_Crab", "crab"),
        ("MarineFish_SeaHorse", "sea horse"),
        ("MarineFish_GhostPipeFish", "ghost pipe fish"),
        ("MarineFish_Butterflyfish", "butterflyfish"),
    ],
)
def test_mas3k_category_excludes_metadata(mas3k_prompt_config, scene, category, expected):
    '''仅输出具体类别，去除分类前缀、场景标记和编号。'''

    image_path = Path(f"MAS_{category}_{scene}_362.jpg")
    assert extract_prompt(image_path, mas3k_prompt_config) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "2511.jpg",
        "MAS_Arthropod_Crab_Unknown_362.jpg",
        "MAS_Arthropod_Crab_Cam.jpg",
        "MAS_Arthropod__Cam_362.jpg",
    ],
)
def test_invalid_mas3k_filename_fails(mas3k_prompt_config, filename):
    '''无效文件名不能作为未清理的类别词传入模型。'''

    with pytest.raises(ValueError, match="Cannot extract prompt"):
        extract_prompt(Path(filename), mas3k_prompt_config)


@pytest.mark.parametrize("normalize", [False, True])
def test_empty_extracted_phrase_fails(mas3k_prompt_config, normalize):
    '''空白替换结果在开启或关闭规范化时都必须报错。'''

    config = replace(
        mas3k_prompt_config, replacement=" \t ", normalize_class_name=normalize
    )
    with pytest.raises(ValueError, match="Cannot extract prompt"):
        extract_prompt(Path("MAS_Arthropod_Crab_Cam_362.jpg"), config)


def test_normalization_collapses_whitespace_after_underscore_replacement():
    '''驼峰拆词与下划线替换后统一大小写和空白。'''

    config = PromptConfig(r"^(.+)_\d+$", r"\1", " \t ", True, True)
    assert extract_prompt(Path("Ghost__PipeFish_1.jpg"), config) == "ghost pipe fish"


def test_normalization_defaults_to_legacy_behavior():
    '''旧构造参数保留大小写、内部空格及原有提取规则。'''

    config = PromptConfig(r"^MAS_(.+)_\d+$", r"\1", " ", True)
    assert config.normalize_class_name is False
    assert (
        extract_prompt(Path("MAS_MarineFish_GhostPipeFish__Cam_1.jpg"), config)
        == "MarineFish GhostPipeFish  Cam"
    )


def test_explicitly_disabled_normalization_keeps_category_token(mas3k_prompt_config):
    '''显式关闭规范化仍可单独使用新的类别提取正则。'''

    config = replace(mas3k_prompt_config, normalize_class_name=False)
    assert (
        extract_prompt(Path("MAS_MarineFish_GhostPipeFish_Cam_1.jpg"), config)
        == "GhostPipeFish"
    )


def test_optional_match_keeps_legacy_filename_fallback():
    '''旧配置允许不匹配时继续返回文件名词组。'''

    config = PromptConfig(r"^MAS_(.+)_\d+$", r"\1", " ", False)
    assert extract_prompt(Path("Ghost_PipeFish.jpg"), config) == "Ghost PipeFish"
