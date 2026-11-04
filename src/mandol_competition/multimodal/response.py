"""Validate public response media without changing pixels, ordering or budgets."""
import base64
import binascii
import io

from PIL import Image,UnidentifiedImageError

from mandol_competition.runtime.contract import OutputContractError
from .protocol import ImagePart

_MIMES={'JPEG':'image/jpeg','PNG':'image/png','WEBP':'image/webp'}
_IMAGE_BYTES=10*1024*1024
_RESPONSE_BYTES=30*1024*1024


def validate_response_media(response):
    total=0
    for hit in response.data:
        if isinstance(hit.content,str):
            continue
        for part in hit.content:
            if not isinstance(part,ImagePart):
                continue
            uri=part.image_url.url
            header,separator,encoded=uri.partition(',')
            if not separator or header not in {'data:'+mime+';base64' for mime in _MIMES.values()}:
                raise OutputContractError('response image must use a supported inline Data URI')
            if len(encoded)>4*((_IMAGE_BYTES+2)//3):
                raise OutputContractError('response image exceeds the wire byte limit')
            try:
                raw=base64.b64decode(encoded,validate=True)
                if not raw or len(raw)>_IMAGE_BYTES:
                    raise OutputContractError('response image exceeds the wire byte limit')
                with Image.open(io.BytesIO(raw)) as image:
                    if _MIMES.get(image.format)!=header[5:-7]:
                        raise OutputContractError('response image format differs from its MIME')
                    image.verify()
            except (ValueError,binascii.Error,UnidentifiedImageError,OSError) as exc:
                raise OutputContractError('response image is not valid encoded media') from exc
            total+=len(raw)
            if total>_RESPONSE_BYTES:
                raise OutputContractError('response media exceeds the wire byte limit')
    return response
