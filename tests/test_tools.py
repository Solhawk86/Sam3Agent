'''批量工具注册与严格 schema 的公开契约。'''

from sam3_agent.tools import (
    AdvanceSegmentationTool,
    ToolContext,
    build_agent_tool_registry,
)


def test_registry_exposes_only_advance_segmentation(tmp_path):
    '''无论候选是否存在，唯一公开工具都是复合决策工具。'''

    context = ToolContext(str(tmp_path / "image.png"), "fish", str(tmp_path))
    registry = build_agent_tool_registry(object(), max_box_tasks_per_round=3)
    definition = registry.definitions(context)[0]["function"]
    assert definition["name"] == "advance_segmentation"
    assert definition["strict"] is True
    assert definition["parameters"]["properties"]["boxes"]["maxItems"] == 3
    context.current_outputs = {"pred_masks": ["encoded"]}
    assert len(registry.definitions(context)) == 1
    assert AdvanceSegmentationTool.__module__.endswith(".advance_segmentation")


def test_all_schema_objects_require_exact_fields():
    '''嵌套对象也必须满足严格工具协议，不允许额外字段。'''

    def inspect(schema):
        '''递归检查 schema 中的对象及数组元素。'''

        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for child in schema["properties"].values():
                inspect(child)
        if "items" in schema:
            inspect(schema["items"])

    inspect(AdvanceSegmentationTool(object()).parameters_schema)


def test_advance_schema_describes_all_decision_parameters():
    '''工具定义为顶层及审核参数提供作用和约束描述。'''

    properties = AdvanceSegmentationTool(object()).parameters_schema["properties"]
    assert all(properties[name].get("description") for name in properties)
    review_properties = properties["review"]["properties"]
    assert all(review_properties[name].get("description") for name in review_properties)
    replacement = review_properties["replace"]["items"]
    assert replacement["description"]
    assert all(
        replacement["properties"][name].get("description")
        for name in replacement["properties"]
    )
