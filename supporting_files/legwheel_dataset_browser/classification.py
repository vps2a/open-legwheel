"""The terrain classes and JSON fields shared by the browser's controls."""


TERRAIN_CLASSES = (
    "Firm Terrain",
    "Loose Terrain",
    "Firm to Loose Transition",
    "Loose to Firm Transition",
)

# Keep these field names in step with the standalone ROS dataset writer.
CLASSIFICATION_FIELDS = ("class_id", "classification_method", "model_id")
