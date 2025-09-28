# from fastapi import FastAPI, Path, Query, HTTPException
# import pandas as pd

# app = FastAPI()
# data = pd.read_json("patients.json")
# patients_json = data.to_dict(orient="records")

# @app.get("/")
# def sayHello():
#     return {"message": "This is an Patient Management System API."}

# @app.get("/view")
# def view():
#     return patients_json

# @app.get("/patient/{pid}")
# def patient(pid: str = Path(..., description = "ID of a particular patient", example = "P001")):
#     for item in patients_json:
#         if item.get("patient_id") == pid:
#             return item
#     # return {"message": "Sorry! The patient you're looking for is not in our DB."}
#     raise HTTPException(status_code=404, detail="patient not found in our DB!")

# @app.get("/showPatients/")
# def showPatients(
#     gender: str = Query(default=None, description="Enter the gender", example="Male"),
#     sort_by: str = Query(default=None, description="Enter the desired sorting order (Asc or Desc)", example="Asc")):

#     if gender is not None and not gender in ["Male", "Female"]:
#         raise HTTPException(404, detail = "gender should be a Male or Female")
#     if sort_by is not None and not sort_by in ["Asc", "Dec"]:
#         raise HTTPException(404, detail = "sorting should be a Asc or Dec")
    
#     res = patients_json

#     if gender is not None:
#         res = [item for item in res if item.get("gender") == gender]

#     if sort_by is not None:
#         if sort_by == "Asc":
#             res = sorted(res, key=lambda item: item["age"])
#         else:
#             res = sorted(res, key=lambda item: -item["age"])

#     return res


# Pydantic

# from pydantic import BaseModel, EmailStr, AnyUrl, Field, field_validator, model_validator, computed_field
# from typing import List, Dict, Optional, Annotated

# class Patient(BaseModel):
#     name: str = Field(max_length = 50)
#     age: Annotated[int, Field(gt = 0, le = 120)]
#     weight: float = Field(gt = 0, strict = True)
#     height: float = Field(gt = 0, strict = True)
#     married: Optional[bool] = False
#     allergies: Annotated[Optional[List[str]], Field(default=None, max_length=5)]
#     email: EmailStr
#     linkedin_url: AnyUrl
#     contact: Dict[str, str]

#     @field_validator("email")
#     @classmethod
#     def email_validator(cls, value):
#         domain = value.split("@")[1]
#         if domain != "sbi.com":
#             raise ValueError("Not a valid domain!!")

#         return value
    
#     @field_validator("age", mode = "before") # Default field_validator mode is "after".
#     @classmethod
#     def age_validator(cls, value):
#         if 0 < value <= 100:
#             return value
#         raise ValueError("Age should be in between 0 and 100!!")
    
#     @model_validator(mode = "after")
#     @classmethod
#     def validate_emergency_contact(cls, model):
#         if model.age > 60 and "emergency" not in model.contact:
#             raise ValueError("Emergency contact must be given if the patient age is more than 60!!")
#         return model

#     @computed_field
#     @property
#     def bmi(self) -> float:
#         return round(self.weight/self.height**2, 2)

# data = {
#     "name": "yogi",
#     "age": 69,
#     "weight": 70.5,
#     "height": 1.72,
#     "email": "abc@sbi.com",
#     "linkedin_url": "https://www.linkedin.com/in/yogi-puvvala",
#     "contact": {
#         "phone": "1234567890",
#         "emergency": "0987654321"
#     }
# }

# patient1 = Patient(**data)

# def insert_data(patient: Patient):
#     print(patient.name)
#     print(patient.age)
#     print(patient.weight)
#     print("BMI:", patient.bmi)
#     print(patient.married)
#     print(patient.allergies)
#     print(patient.email)
#     print(patient.linkedin_url)
#     print(patient.contact)

# insert_data(patient1)

from pydantic import BaseModel

class Address(BaseModel):

    city: str
    state: str
    pincode: str

class Patient(BaseModel):

    name: str
    age: int
    gender: str
    address: Address

data = {
    "name": "Yogi",
    "age": 19,
    "gender": "male",
    "address": {
        "city": "visakhapatnam",
        "state": "AP",
        "pincode": "530011"
    }
}

patient1 = Patient(**data)

# def insert_data(patient: Patient):
#     print(patient.name)
#     print(patient.age)
#     print(patient.gender)
#     print(patient.address.city)
#     print(patient.address.state)
#     print(patient.address.pincode)

# insert_data(patient1)

details = patient1.model_dump(include={"name", "age"})
# details = patient1.model_dump(exclude = {"address": ["city"]})
print(details)

