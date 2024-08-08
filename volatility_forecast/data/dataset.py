from .base import DataSet, Field


class PriceVolume(DataSet):
    OPEN = Field(dtype=float)
    HIGH = Field(dtype=float)
    LOW = Field(dtype=float)
    CLOSE = Field(dtype=float)
    VOLUME = Field(dtype=float)
