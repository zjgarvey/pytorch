# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def hipdnn_available():
    """Whether the build includes hipDNN support (USE_HIPDNN). When False,
    setting `torch.backends.miopen.use_hipdnn = True` is a no-op: convolutions
    continue to dispatch to MIOpen.
    """
    return torch._C._has_hipdnn


def set_flags(
    _immediate=None,
    _use_hipdnn=None,
):
    orig_flags = (
        torch._C._get_miopen_immediate(),
        torch._C._get_miopen_use_hipdnn(),
    )
    if _immediate is not None:
        torch._C._set_miopen_immediate(_immediate)
    if _use_hipdnn is not None:
        torch._C._set_miopen_use_hipdnn(_use_hipdnn)
    return orig_flags


@contextmanager
def flags(
    immediate=False,
    use_hipdnn=None,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            immediate,
            use_hipdnn,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.miopen.immediate = True
#   torch.backends.miopen.use_hipdnn = True


class MiopenModule(PropModule):
    immediate = ContextProp(
        torch._C._get_miopen_immediate, torch._C._set_miopen_immediate
    )
    # Transitional toggle: when True, MIOpen convolution dispatch routes to
    # hipDNN instead. Builds without USE_HIPDNN ignore this flag (see
    # hipdnn_available()). Will be removed once hipDNN fully subsumes MIOpen.
    use_hipdnn = ContextProp(
        torch._C._get_miopen_use_hipdnn, torch._C._set_miopen_use_hipdnn
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MiopenModule(sys.modules[__name__], __name__)

# Add type annotations for the replaced module
immediate: bool
use_hipdnn: bool
