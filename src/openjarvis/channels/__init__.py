"""Channel abstraction for multi-platform messaging."""

import importlib

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)

# Trigger registration of built-in channels.
# Each module uses @ChannelRegistry.register() — importing is sufficient.
# Pruned for JARVIS personal: only channels relevant to Pau are loaded.
_CHANNEL_MODULES = [
    "telegram",
    "discord_channel",
    "slack",
    "webhook",
    "email_channel",
    "whatsapp",
    "signal_channel",
    "google_chat",
    "webchat",
    "teams",
    "matrix_channel",
    "whatsapp_baileys",
    "messenger_channel",
    "twitter",
    "gmail",
]

for _mod in _CHANNEL_MODULES:
    try:
        importlib.import_module(f".{_mod}", __name__)
    except ImportError:
        pass

__all__ = [
    "BaseChannel",
    "ChannelHandler",
    "ChannelMessage",
    "ChannelStatus",
]
