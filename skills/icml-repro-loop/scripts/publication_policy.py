"""Publishing-account policy shared by scheduling and Hub authority."""


ALLOWED_SPACE_OWNERS = frozenset({"wrice"})


def space_owner(space_id: object) -> str:
    """Return the owner of one exact ``owner/name`` Space identifier."""
    if type(space_id) is not str:
        raise ValueError("space_id")
    owner, separator, name = space_id.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise ValueError("space_id")
    return owner
