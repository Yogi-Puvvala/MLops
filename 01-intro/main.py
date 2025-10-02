# Get Request:

from fastapi import FastAPI, Path, Query, HTTPException
import pandas as pd
import json

app = FastAPI()

with open("patients.json", "r") as file:
    data = json.load(file)

@app.get("/")
def sayHello():
    return {"message": "This is an Patient Management System API."}

@app.get("/view")
def view():
    return data

@app.get("/patient/{pid}")
def patient(pid: str = Path(..., description = "ID of a particular patient", example = "P001")):
    for item in data:
        if item.get("pid") == pid:
            return item
    # return {"message": "Sorry! The patient you're looking for is not in our DB."}
    raise HTTPException(status_code=404, detail="patient not found in our DB!")

@app.get("/showPatients/")
def showPatients(
    gender: str = Query(default=None, description="Enter the gender", example="Male"),
    sort_by: str = Query(default=None, description="Enter the desired sorting order (Asc or Desc)", example="Asc")):

    if gender is not None and not gender in ["Male", "Female"]:
        raise HTTPException(404, detail = "gender should be a Male or Female")
    if sort_by is not None and not sort_by in ["Asc", "Dec"]:
        raise HTTPException(404, detail = "sorting should be a Asc or Dec")
    
    res = data

    if gender is not None:
        res = [item for item in res if item.get("gender") == gender]

    if sort_by is not None:
        if sort_by == "Asc":
            res = sorted(res, key=lambda item: item["age"])
        else:
            res = sorted(res, key=lambda item: -item["age"])

    return res


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

# from pydantic import BaseModel

# class Address(BaseModel):

#     city: str
#     state: str
#     pincode: str

# class Patient(BaseModel):

#     name: str
#     age: int
#     gender: str
#     address: Address

# data = {
#     "name": "Yogi",
#     "age": 19,
#     "gender": "male",
#     "address": {
#         "city": "visakhapatnam",
#         "state": "AP",
#         "pincode": "530011"
#     }
# }

# patient1 = Patient(**data)

# def insert_data(patient: Patient):
#     print(patient.name)
#     print(patient.age)
#     print(patient.gender)
#     print(patient.address.city)
#     print(patient.address.state)
#     print(patient.address.pincode)

# insert_data(patient1)

# details = patient1.model_dump(include = {"name", "age"})
# # details = patient1.model_dump(exclude = {"address": ["city"]})
# print(details)


# Post Request

from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal

class Patient(BaseModel):
    pid: Annotated[str, Field(description="Patient ID", examples=["P001"])]
    name: Annotated[str, Field(description="Name of the patient", examples=["Yogi"])]
    city: Annotated[str, Field(description="Name of the city", examples=["Visakhapatnam", "Guntur"])]
    age: Annotated[int, Field(gt=0, description="Age of the patient", examples=[19, 27, 45])]
    gender: Annotated[Literal["Male", "Female", "Others"], Field(description="Gender of the patient")]
    height: Annotated[float, Field(description="Height of the patient", examples=[1.72], gt=0)]
    weight: Annotated[float, Field(description="Weight of the patient", examples=[70.5], gt=0)]

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight/self.height**2, 2)
    
    @computed_field
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return "Under Weight"
        elif self.bmi < 30:
            return "Normal"
        else:
            return "Over Weight"
        
@app.post("/create")
def create_patient(patient: Patient):

    # Check whether the patient already exists
    for item in data:
        if item["pid"] == patient.pid:
            raise HTTPException(status_code=400, detail="The patient already exists in our database!")
        
    # Add patient to the database
    data.append(patient.model_dump())

    # Save the data to the JSON file
    with open("patients.json", "w") as file:
        json.dump(data, file)

    return {"message": "Patient created successfully!"}


# Put Request

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class PatientUpdate(BaseModel):
    pid: Annotated[Optional[str], Field(description="Patient ID", examples=["P001"], default=None)]
    name: Annotated[Optional[str], Field(description="Name of the patient", examples=["Yogi"], default=None)]
    city: Annotated[Optional[str], Field(description="Name of the city", examples=["Visakhapatnam", "Guntur"], default=None)]
    age: Annotated[Optional[int], Field(gt=0, description="Age of the patient", examples=[19, 27, 45], default=None)]
    gender: Annotated[Optional[Literal["Male", "Female", "Others"]], Field(description="Gender of the patient", default=None)]
    height: Annotated[Optional[float], Field(description="Height of the patient", examples=[1.72], gt=0, default=None)]
    weight: Annotated[Optional[float], Field(description="Weight of the patient", examples=[70.5], gt=0, default=None)]

@app.put("/edit/{pid}")
def update_patient(pid: str, patient_update: PatientUpdate):
    req_data = {}

    for item in data:
        if item["pid"] == pid:
            req_data = item.copy()  # safer than reference

    # Checking whether the patient exists or not
    if not req_data:
        raise HTTPException(status_code=404, detail="The patient doesn't exist in the db.")
    
    # Combining the updated and previous values
    updated_data = patient_update.model_dump()
    req_data.update(updated_data)

    # Placing the updated row in the db
    for i, item in enumerate(data):
        if item["pid"] == req_data["pid"]:
            data[i] = req_data
            break

    # Updating the JSON file
    with open("patients.json", "w") as file:
        json.dump(data, file, indent=4)

    return {"message": "Patient info updated successfully."}


# Delete Request

@app.delete("/remove/{pid}")
def remove_patient(pid: str):
    req_data = {}

    for item in data:
        if item["pid"] == pid:
            req_data = item
            break

    # Checking whether the data exist in db or not
    if not req_data:
        raise HTTPException(status_code=404, detail="Patient doesn't exist in db!!")
    
    # Removing the data
    data.remove(req_data)

    # Updating the Json file
    with open("patients.json", "w") as file:
        json.dump(data, file, indent=4)

    return {"message": "The specified patient info successfully removed from db."}


# Serving ML Models with FastAPI

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import pickle
import pandas

class Person(BaseModel):

    bmi: Annotated[float, Field(gt=0, description="Body-mass index of the person", examples=[21.92])]
    age_group: Annotated[Literal["young", "adult", "middle_aged", "senior"], Field(description="Age group of the person", examples=["young", "senior"])]
    lifestyle_risk: Annotated[Literal["low", "medium", "high"], Field(description="Lifestyle risk of the person", examples = ["low", "high"])]
    city_tier: Annotated[Literal["1", "2", "3"], Field(description="City-tier of the person", examples=["1", "3"])]
    income_lpa: Annotated[float, Field(description="Income (in LPA) of the person", examples=[14.54, 36.69])]
    occupation: Annotated[str, Field(description="Occupation of the person", examples=["freelancer", "retired"])]

@app.post("/check/")
def check_person(person: Person):

    # Loading Model
    with open("insurance.pkl", "rb") as file:
        model = pickle.load(file)

    # Converting the person's json into dataframe
    df = pd.DataFrame(person.model_dump(), index = [0])
    res = model.predict(df)

    return {"Insurance Premium Category": res[0]}

