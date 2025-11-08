"""
MedSynth Platform - User Schemas
Pydantic models for API validation
"""
from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from app.models.user import UserRole, SubscriptionTier


# User Schemas
class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User schema with database fields"""
    id: int
    is_active: bool
    is_verified: bool
    is_superuser: bool
    role: UserRole
    subscription_tier: SubscriptionTier
    organization_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserResponse(UserInDB):
    """User response schema"""
    pass


# Organization Schemas
class OrganizationBase(BaseModel):
    """Base organization schema"""
    name: str
    slug: str
    description: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None


class OrganizationCreate(OrganizationBase):
    """Schema for creating a new organization"""
    pass


class OrganizationUpdate(BaseModel):
    """Schema for updating organization"""
    name: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    is_active: Optional[bool] = None
    subscription_tier: Optional[SubscriptionTier] = None
    hipaa_compliant: Optional[bool] = None
    gdpr_compliant: Optional[bool] = None


class OrganizationInDB(OrganizationBase):
    """Organization schema with database fields"""
    id: int
    is_active: bool
    subscription_tier: SubscriptionTier
    hipaa_compliant: bool
    gdpr_compliant: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class OrganizationResponse(OrganizationInDB):
    """Organization response schema"""
    pass


# Authentication Schemas
class Token(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Token payload schema"""
    sub: Optional[int] = None
    exp: Optional[int] = None
    type: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str


class PasswordChange(BaseModel):
    """Password change schema"""
    current_password: str
    new_password: str

    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
