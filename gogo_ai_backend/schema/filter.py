from typing import Optional

from pydantic import BaseModel

class filtered_result(BaseModel):
    id: str
    boardId: Optional[int] = None 
    commentId: Optional[int] = None  

    class Config:
        exclude_none = True