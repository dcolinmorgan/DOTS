curl -X 'GET' \
  'https://den.louie.ai/api/account' \
  -H 'Authorization: Bearer my_valid_token'

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.oauth2 import OAuth2PasswordBearer

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != "my_valid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.get("/account", dependencies=[Depends(get_current_user)])
async def read_account():
    return {"account": "account details"}
