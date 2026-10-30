from pydantic import BaseModel
from pydantic import Field


class VLMResponse(BaseModel):
    text: str = Field(description="Generated text from the image and prompt.")


def generate_response_format(model: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }
