from pydantic import BaseModel, Field
from typing import Optional

class InsuranceInput(BaseModel):
    # Definimos los campos exactos que espera tu preprocesador
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., ge=18, le=100, example=44)
    HasDrivingLicense: int = Field(..., example=1)
    RegionID: float = Field(..., example=28.0)
    Switch: float = Field(..., example=0.0) # Antes llamada 'Switch' en tu CSV
    VehicleAge: str = Field(..., example="> 2 Years")
    PastAccident: str = Field(..., example="Yes")
    AnnualPremium: str = Field(..., example="£40454.0") # String por el símbolo £
    SalesChannelID: float = Field(..., example=26.0)
    DaysSinceCreated: int = Field(..., example=217)
    
    # Configuración extra para documentación en Swagger UI
    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 44,
                "HasDrivingLicense": 1,
                "RegionID": 28.0,
                "Switch": 0.0,
                "VehicleAge": "> 2 Years",
                "PastAccident": "Yes",
                "AnnualPremium": "£40454.0",
                "SalesChannelID": 26.0,
                "DaysSinceCreated": 217
            }
        }

class PredictionOutput(BaseModel):
    prediction: int       # 0 (No Venta) o 1 (Venta)
    probability: float    # Probabilidad bruta (0.75)
    threshold_used: float # El corte que usamos (ej. 0.67)
    model_version: str    # ID del experimento para trazabilidad