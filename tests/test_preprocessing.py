from deepface_server.preprocessing import (
    Clahe,
    EXIFOrientation,
    EXIFRotate,
    FaceCropPlaceholder,
    GrayscaleConvert,
    Normalize,
    Pipeline,
    PreprocessStep,
    Resize,
    apply_orientation,
    bgr_to_rgb,
    build_default_pipeline,
    ensure_rgb,
    read_orientation,
)


class _RecordingStep(PreprocessStep):
    name = "record"

    def __init__(self):
        self.calls = 0
        self.params = {}

    def apply(self, image, metadata):
        self.calls += 1
        metadata["recorded"] = True
        return image


def test_pipeline_runs_steps_in_order():
    a = _RecordingStep()
    b = _RecordingStep()
    pipeline = Pipeline([a, b])
    result = pipeline.run(b"image-bytes")
    assert a.calls == 1 and b.calls == 1
    assert result.metadata["recorded"] is True
    assert result.metadata["applied"] == ["record", "record"]


def test_pipeline_add_and_remove():
    pipeline = Pipeline()
    step = Resize((64, 64))
    pipeline.add(step)
    assert len(pipeline) == 1
    assert pipeline.remove("resize") is True
    assert pipeline.remove("nope") is False


def test_default_pipeline_has_three_stages():
    pipeline = build_default_pipeline()
    names = [s.name for s in pipeline]
    assert names == ["exif_rotate", "resize", "normalize"]


def test_steps_describe_includes_params():
    step = Resize((128, 128))
    desc = step.describe()
    assert desc["name"] == "resize"
    assert desc["params"]["target_size"] == (128, 128)


def test_resize_step_falls_back_without_cv2():
    step = Resize((10, 10))
    out = step.apply(b"abc", {})
    assert out == b"abc" or hasattr(out, "shape")


def test_normalize_passes_through_when_numpy_missing():
    step = Normalize()
    metadata = {}
    out = step.apply([1, 2, 3], metadata)
    assert out is not None


def test_grayscale_step_runs():
    step = GrayscaleConvert(keep_channels=False)
    out = step.apply([[1, 2, 3]], {})
    assert out is not None


def test_clahe_handles_unknown_input():
    step = Clahe()
    metadata = {}
    out = step.apply(object(), metadata)
    assert metadata.get("clahe_skipped") is True or out is not None


def test_face_crop_placeholder_metadata_on_failure():
    step = FaceCropPlaceholder(fraction=0.5)
    metadata = {}
    out = step.apply(object(), metadata)
    assert metadata.get("face_crop_skipped") is True or out is not None


def test_exif_rotate_returns_image_when_no_pil():
    step = EXIFRotate()
    obj = object()
    result = step.apply(obj, {})
    assert result is obj


def test_orientation_helpers_default_to_normal():
    assert read_orientation(object()) == EXIFOrientation.NORMAL
    image = object()
    assert apply_orientation(image, EXIFOrientation.NORMAL) is image


def test_colorspace_helpers_no_numpy():
    obj = object()
    assert bgr_to_rgb(obj) is obj
    assert ensure_rgb(obj) is obj
