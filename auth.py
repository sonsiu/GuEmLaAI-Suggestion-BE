from fastapi import Request, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ISSUER = os.getenv("JWT_ISSUER")
AUDIENCE = os.getenv("JWT_AUDIENCE")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def authorize_user(token: str = Depends(oauth2_scheme)):
    # print("Authorizing user with token:", token)
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["exp", "sub"]},
        )

        # Extract claims
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("Role")

        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing subject")

        return {"user_id": user_id, "email": email, "role": role}

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
        )
