def test_package_is_importable() -> None:
    import myvllm

    assert myvllm.__all__ == [
        "CacheConfig",
        "ModelConfig",
        "ParallelConfig",
        "RequestOutput",
        "SamplingParams",
        "SchedulerConfig",
    ]

    assert myvllm.SamplingParams(max_tokens=1).max_tokens == 1
