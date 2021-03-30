from sqlalchemy.orm import Session

from . import models


def get_value(db: Session, key: str):
    x = db.query(models.Hash).filter(models.Hash.key == key).first()
    if x is None:
        return x 
    else:
        return x.value


def set_value(db: Session, key: str, value: str):
    db_datum = db.query(models.Hash).filter(models.Hash.key == key).one_or_none()
    if db_datum is None:
        db_datum = models.Hash(key=key, value=value)
    else:
        db_datum.value = value
    db.add(db_datum)
    db.commit()
    db.refresh(db_datum)
    return db_datum

def delete_value(db: Session, key: str):
    db_datum = db.query(models.Hash).filter(models.Hash.key == key).one_or_none()
    db.delete(db_datum)
    db.commit()
    return "Deleted"