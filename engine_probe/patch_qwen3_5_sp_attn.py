#!/usr/bin/env python3
"""Patch an installed verl so Qwen3.5 Ulysses SP works with non-flash attention backends.

verl only installs the Ulysses all-to-all into
`transformers.integrations.flash_attention._flash_attention_forward`. With
`attn_implementation=sdpa` that hook is never reached, so every full_attention layer
attends only inside its own rank's contiguous 1/sp slice of the sequence.

Idempotent. `--check` verifies without writing.
"""

import argparse
import base64
import importlib.util
import os
import sys

FN_B64 = "ZGVmIF9wYWNrZWRfc2VnbWVudF9ib3VuZHMoY3Vfc2VxbGVuc19jcHUsIGN1X3NlcWxlbnMsIHRvdGFsX2xlbjogaW50KSAtPiBsaXN0W2ludF06CiAgICAiIiJCb3VuZGFyaWVzIG9mIHRoZSBwYWNrZWQgc2FtcGxlcyBpbnNpZGUgYSBnYXRoZXJlZCAoZ2xvYmFsKSBzZXF1ZW5jZSBvZiBgYHRvdGFsX2xlbmBgLgoKICAgIGBgY3Vfc2VxbGVuc2BgIGlzIHRoZSB1bnNsaWNlZCBvZmZzZXRzIHRlbnNvciB0aGUgZW5naW5lIHBhc3NlcyBkb3duLCBhbHJlYWR5IGV4dGVuZGVkCiAgICB3aXRoIHRoZSBzZXF1ZW5jZS1wYXJhbGxlbCBwYWQgd2hlbiB0aGVyZSBpcyBvbmUgKHNlZSBgYHRyYW5zZm9ybWVyX2ltcGwucHlgYCkuIFByZWZlcgogICAgdGhlIGNwdSBjb3B5IC0tIHJlYWRpbmcgdGhlIGN1ZGEgb25lIGJhY2sgd291bGQgc3luYyBvbiBldmVyeSBsYXllci4KICAgICIiIgogICAgYm91bmRzID0gY3Vfc2VxbGVuc19jcHUgaWYgY3Vfc2VxbGVuc19jcHUgaXMgbm90IE5vbmUgZWxzZSBjdV9zZXFsZW5zCiAgICBpZiBib3VuZHMgaXMgTm9uZToKICAgICAgICByZXR1cm4gWzAsIHRvdGFsX2xlbl0KCiAgICByYXcgPSBbaW50KHgpIGZvciB4IGluIGJvdW5kcy50b2xpc3QoKV0KICAgIGlmIHJhd1swXSAhPSAwIG9yIHJhd1stMV0gPiB0b3RhbF9sZW46CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmImN1X3NlcWxlbnMge3Jhd1swXX0uLntyYXdbLTFdfSBkb2VzIG5vdCBkZXNjcmliZSBhIHNlcXVlbmNlIG9mIHt0b3RhbF9sZW59IHRva2VucyIpCgogICAgYm91bmRzID0gWzBdCiAgICBmb3Igb2Zmc2V0IGluIHJhd1sxOl06CiAgICAgICAgaWYgb2Zmc2V0ID4gYm91bmRzWy0xXTogICMgZHJvcCBlbXB0eSBzYW1wbGVzCiAgICAgICAgICAgIGJvdW5kcy5hcHBlbmQob2Zmc2V0KQogICAgaWYgYm91bmRzWy0xXSA8IHRvdGFsX2xlbjoKICAgICAgICBib3VuZHMuYXBwZW5kKHRvdGFsX2xlbikgICMgdHJhaWxpbmcgc2VxdWVuY2UtcGFyYWxsZWwgcGFkZGluZwogICAgcmV0dXJuIGJvdW5kcwoKCmRlZiBxd2VuM181X2F0dG5fZm9yd2FyZCgKICAgIHNlbGYsCiAgICBoaWRkZW5fc3RhdGVzOiB0b3JjaC5UZW5zb3IsCiAgICBwb3NpdGlvbl9lbWJlZGRpbmdzOiB0dXBsZVt0b3JjaC5UZW5zb3IsIHRvcmNoLlRlbnNvcl0sCiAgICBhdHRlbnRpb25fbWFzazogT3B0aW9uYWxbdG9yY2guVGVuc29yXSA9IE5vbmUsCiAgICBwYXN0X2tleV92YWx1ZXM9Tm9uZSwKICAgICoqa3dhcmdzLAopOgogICAgIiIiQmFja2VuZC1hZ25vc3RpYyBVbHlzc2VzIFNQIGZvciBRd2VuMy41IGZ1bGwtYXR0ZW50aW9uIGxheWVycy4KCiAgICBUaGUgZ2VuZXJpYyBob29rIGluIGBgbW9ua2V5X3BhdGNoLnB5YGAgb25seSByZXBsYWNlcwogICAgYGB0cmFuc2Zvcm1lcnMuaW50ZWdyYXRpb25zLmZsYXNoX2F0dGVudGlvbi5fZmxhc2hfYXR0ZW50aW9uX2ZvcndhcmRgYCwgc28gaXQgaXMgZGVhZAogICAgY29kZSB3aGVuZXZlciBgYGF0dG5faW1wbGVtZW50YXRpb25gYCBpcyBub3QgZmxhc2ggYXR0ZW50aW9uIChlLmcuIGBgc2RwYWBgKS4gV2l0aG91dAogICAgYW4gYWxsLXRvLWFsbCBlYWNoIHJhbmsgYXR0ZW5kcyBvbmx5IGluc2lkZSBpdHMgb3duIGNvbnRpZ3VvdXMgMS9zcCBzbGljZSBvZiB0aGUKICAgIHNlcXVlbmNlIC0tIGEgc2lsZW50IGNvcnJlY3RuZXNzIGJ1Zywgbm90IGEgcHJlY2lzaW9uIG9uZS4KCiAgICBUaGlzIG1pcnJvcnMgYGB2ZXJsL21vZGVscy90cmFuc2Zvcm1lcnMvcXdlbjIucHlgYDogZ2F0aGVyIHRoZSBzZXF1ZW5jZSAvIHNjYXR0ZXIgdGhlCiAgICBoZWFkcyBhcm91bmQgd2hpY2hldmVyIGF0dGVudGlvbiBpbnRlcmZhY2UgdGhlIGNvbmZpZyBzZWxlY3RzLiBSb1BFIGlzIGFwcGxpZWQgYmVmb3JlCiAgICB0aGUgYWxsLXRvLWFsbCwgb24gdGhlIGxvY2FsIHNoYXJkLCBiZWNhdXNlIGNvcy9zaW4gYXJlIGJ1aWx0IGZyb20gdGhlIGxvY2FsIChidXQKICAgIGdsb2JhbGx5LXZhbHVlZCkgcG9zaXRpb24gaWRzLgoKICAgIFBhY2tpbmcgaXMgaGFuZGxlZCBieSBsb29waW5nIG92ZXIgdGhlICpnbG9iYWwqIGBgY3Vfc2VxbGVuc2BgIHNlZ21lbnRzIHdpdGggYSBjYXVzYWwsCiAgICBtYXNrLWZyZWUgY2FsbCBwZXIgc2VnbWVudC4gSEYgYnVpbGRzIGl0cyBvd24gYmxvY2stY2F1c2FsIG1hc2sgZnJvbSB0aGUgbG9jYWwgc2hhcmQncwogICAgcG9zaXRpb24gaWRzLCB3aGljaCBkZXNjcmliZXMgdGhlIHdyb25nIHJvd3MvY29sdW1ucyBvbmNlIHRoZSBzZXF1ZW5jZSBpcyBnYXRoZXJlZCwgYW5kCiAgICB0aGUgZ2xvYmFsIGRlbnNlIGVxdWl2YWxlbnQgd291bGQgYmUgYGB0b3RhbF9ubnoqKjJgYCAofjE2IEdCIGF0IDEyNmsgdG9rZW5zKS4KCiAgICBGbGFzaC1hdHRlbnRpb24gYmFja2VuZHMgYXJlIGxlZnQgYWxvbmU6IGBgbW9ua2V5X3BhdGNoYGAgYWxzbyByZXBsYWNlcwogICAgYGBfZmxhc2hfYXR0ZW50aW9uX2ZvcndhcmRgYCwgd2hpY2ggYWxyZWFkeSBkb2VzIHRoZSBhbGwtdG8tYWxsIHRoZXJlLgogICAgIiIiCiAgICBmcm9tIHRyYW5zZm9ybWVycy5tb2RlbGluZ191dGlscyBpbXBvcnQgQUxMX0FUVEVOVElPTl9GVU5DVElPTlMKICAgIGZyb20gdHJhbnNmb3JtZXJzLm1vZGVscy5xd2VuM181Lm1vZGVsaW5nX3F3ZW4zXzUgaW1wb3J0ICgKICAgICAgICBhcHBseV9yb3RhcnlfcG9zX2VtYiwKICAgICAgICBlYWdlcl9hdHRlbnRpb25fZm9yd2FyZCwKICAgICAgICByZXBlYXRfa3YsCiAgICApCgogICAgZnJvbSB2ZXJsLnV0aWxzLnVseXNzZXMgaW1wb3J0IGdhdGhlcl9oZWFkc19zY2F0dGVyX3NlcSwgZ2F0aGVyX3NlcV9zY2F0dGVyX2hlYWRzCgogICAgY3Vfc2VxbGVucyA9IGt3YXJncy5wb3AoImN1X3NlcWxlbnMiLCBOb25lKQogICAgY3Vfc2VxbGVuc19jcHUgPSBrd2FyZ3MucG9wKCJjdV9zZXFsZW5zX2NwdSIsIE5vbmUpCgogICAgaW5wdXRfc2hhcGUgPSBoaWRkZW5fc3RhdGVzLnNoYXBlWzotMV0KICAgIGhpZGRlbl9zaGFwZSA9ICgqaW5wdXRfc2hhcGUsIC0xLCBzZWxmLmhlYWRfZGltKQoKICAgIHF1ZXJ5X3N0YXRlcywgZ2F0ZSA9IHRvcmNoLmNodW5rKHNlbGYucV9wcm9qKGhpZGRlbl9zdGF0ZXMpLnZpZXcoKmlucHV0X3NoYXBlLCAtMSwgc2VsZi5oZWFkX2RpbSAqIDIpLCAyLCBkaW09LTEpCiAgICBnYXRlID0gZ2F0ZS5yZXNoYXBlKCppbnB1dF9zaGFwZSwgLTEpCgogICAgcXVlcnlfc3RhdGVzID0gc2VsZi5xX25vcm0ocXVlcnlfc3RhdGVzLnZpZXcoaGlkZGVuX3NoYXBlKSkudHJhbnNwb3NlKDEsIDIpCiAgICBrZXlfc3RhdGVzID0gc2VsZi5rX25vcm0oc2VsZi5rX3Byb2ooaGlkZGVuX3N0YXRlcykudmlldyhoaWRkZW5fc2hhcGUpKS50cmFuc3Bvc2UoMSwgMikKICAgIHZhbHVlX3N0YXRlcyA9IHNlbGYudl9wcm9qKGhpZGRlbl9zdGF0ZXMpLnZpZXcoaGlkZGVuX3NoYXBlKS50cmFuc3Bvc2UoMSwgMikKCiAgICBjb3MsIHNpbiA9IHBvc2l0aW9uX2VtYmVkZGluZ3MKICAgIHF1ZXJ5X3N0YXRlcywga2V5X3N0YXRlcyA9IGFwcGx5X3JvdGFyeV9wb3NfZW1iKHF1ZXJ5X3N0YXRlcywga2V5X3N0YXRlcywgY29zLCBzaW4pCgogICAgaWYgcGFzdF9rZXlfdmFsdWVzIGlzIG5vdCBOb25lOgogICAgICAgIGtleV9zdGF0ZXMsIHZhbHVlX3N0YXRlcyA9IHBhc3Rfa2V5X3ZhbHVlcy51cGRhdGUoa2V5X3N0YXRlcywgdmFsdWVfc3RhdGVzLCBzZWxmLmxheWVyX2lkeCkKCiAgICBhdHRuX2ltcGwgPSBzZWxmLmNvbmZpZy5fYXR0bl9pbXBsZW1lbnRhdGlvbgogICAgdWx5c3Nlc19zcF9zaXplID0gZ2V0X3VseXNzZXNfc2VxdWVuY2VfcGFyYWxsZWxfd29ybGRfc2l6ZSgpCiAgICAjIGZsYXNoIGJhY2tlbmRzIGdldCB0aGVpciBhbGwtdG8tYWxsIGZyb20gdGhlIGBfZmxhc2hfYXR0ZW50aW9uX2ZvcndhcmRgIGhvb2sKICAgIGRvX2FsbF90b19hbGwgPSB1bHlzc2VzX3NwX3NpemUgPiAxIGFuZCAiZmxhc2giIG5vdCBpbiBzdHIoYXR0bl9pbXBsKQoKICAgIG51bV9rZXlfdmFsdWVfZ3JvdXBzID0gc2VsZi5udW1fa2V5X3ZhbHVlX2dyb3VwcwogICAgaWYgZG9fYWxsX3RvX2FsbDoKICAgICAgICBpZiBhdHRlbnRpb25fbWFzayBpcyBub3QgTm9uZSBhbmQgY3Vfc2VxbGVucyBpcyBOb25lIGFuZCBjdV9zZXFsZW5zX2NwdSBpcyBOb25lOgogICAgICAgICAgICAjIGEgbWFzayBtZWFucyB0aGUgYmF0Y2ggaXMgcGFja2VkIChvciBwYWRkZWQpOyB3aXRob3V0IHRoZSBnbG9iYWwgYm91bmRhcmllcyB0aGUKICAgICAgICAgICAgIyBzZWdtZW50IGxvb3AgYmVsb3cgd291bGQgbGV0IHNhbXBsZSBpIGF0dGVuZCB0byBzYW1wbGUgaS0xLiBGYWlsIGxvdWRseS4KICAgICAgICAgICAgcmFpc2UgTm90SW1wbGVtZW50ZWRFcnJvcigKICAgICAgICAgICAgICAgICJRd2VuMy41IFVseXNzZXMgU1AgbmVlZHMgYGN1X3NlcWxlbnNgIHRvIHJlYnVpbGQgdGhlIHBhY2tpbmcgYm91bmRhcmllcyBhZnRlciB0aGUgIgogICAgICAgICAgICAgICAgZiJhbGwtdG8tYWxsLCBidXQgZ290IG9ubHkgYSBsb2NhbC1zaGFyZCBtYXNrIG9mIHNoYXBlIHt0dXBsZShhdHRlbnRpb25fbWFzay5zaGFwZSl9LiIKICAgICAgICAgICAgKQogICAgICAgICMgcmVwZWF0IGt2IHNvIHRoZSBoZWFkIGNvdW50IGlzIGRpdmlzaWJsZSBieSBzcCAoc2FtZSBydWxlIGFzIF91bHlzc2VzX2ZsYXNoX2F0dGVudGlvbl9mb3J3YXJkKQogICAgICAgIHJlcGVhdHMgPSBtYXgodWx5c3Nlc19zcF9zaXplIC8vIGtleV9zdGF0ZXMuc2l6ZSgxKSwgMSkKICAgICAgICBrZXlfc3RhdGVzID0gcmVwZWF0X2t2KGtleV9zdGF0ZXMsIHJlcGVhdHMpCiAgICAgICAgdmFsdWVfc3RhdGVzID0gcmVwZWF0X2t2KHZhbHVlX3N0YXRlcywgcmVwZWF0cykKCiAgICAgICAgIyAoYnN6LCBuX2hlYWQsIHNlcV9sZW4vc3AsIGhlYWRfZGltKSAtPiAoYnN6LCBuX2hlYWQvc3AsIHNlcV9sZW4sIGhlYWRfZGltKQogICAgICAgIHF1ZXJ5X3N0YXRlcyA9IGdhdGhlcl9zZXFfc2NhdHRlcl9oZWFkcyhxdWVyeV9zdGF0ZXMsIHNlcV9kaW09MiwgaGVhZF9kaW09MSkKICAgICAgICBrZXlfc3RhdGVzID0gZ2F0aGVyX3NlcV9zY2F0dGVyX2hlYWRzKGtleV9zdGF0ZXMsIHNlcV9kaW09MiwgaGVhZF9kaW09MSkKICAgICAgICB2YWx1ZV9zdGF0ZXMgPSBnYXRoZXJfc2VxX3NjYXR0ZXJfaGVhZHModmFsdWVfc3RhdGVzLCBzZXFfZGltPTIsIGhlYWRfZGltPTEpCgogICAgICAgICMgc2RwYS9lYWdlciByZS1leHBhbmQga3YgdXNpbmcgbW9kdWxlLm51bV9rZXlfdmFsdWVfZ3JvdXBzOyBhZnRlciB0aGUgcmVwZWF0IGFuZAogICAgICAgICMgc2NhdHRlciBhYm92ZSB0aGF0IHJhdGlvIGlzIG5vIGxvbmdlciB0aGUgb25lIHN0b3JlZCBvbiB0aGUgbW9kdWxlLgogICAgICAgIG51bV9rZXlfdmFsdWVfZ3JvdXBzID0gcXVlcnlfc3RhdGVzLnNpemUoMSkgLy8ga2V5X3N0YXRlcy5zaXplKDEpCgogICAgYXR0ZW50aW9uX2ludGVyZmFjZSA9IEFMTF9BVFRFTlRJT05fRlVOQ1RJT05TLmdldF9pbnRlcmZhY2UoYXR0bl9pbXBsLCBlYWdlcl9hdHRlbnRpb25fZm9yd2FyZCkKICAgIGF0dG5fY29tbW9uID0gZGljdCgKICAgICAgICBkcm9wb3V0PTAuMCBpZiBub3Qgc2VsZi50cmFpbmluZyBlbHNlIHNlbGYuYXR0ZW50aW9uX2Ryb3BvdXQsCiAgICAgICAgc2NhbGluZz1zZWxmLnNjYWxpbmcsCiAgICAgICAgKiprd2FyZ3MsCiAgICApCgogICAgb3JpZ19ncm91cHMgPSBzZWxmLm51bV9rZXlfdmFsdWVfZ3JvdXBzCiAgICBzZWxmLm51bV9rZXlfdmFsdWVfZ3JvdXBzID0gbnVtX2tleV92YWx1ZV9ncm91cHMKICAgIHRyeToKICAgICAgICBpZiBkb19hbGxfdG9fYWxsOgogICAgICAgICAgICBzZWdtZW50cyA9IF9wYWNrZWRfc2VnbWVudF9ib3VuZHMoY3Vfc2VxbGVuc19jcHUsIGN1X3NlcWxlbnMsIHF1ZXJ5X3N0YXRlcy5zaXplKDIpKQogICAgICAgICAgICBvdXRwdXRzID0gW10KICAgICAgICAgICAgYXR0bl93ZWlnaHRzID0gTm9uZQogICAgICAgICAgICBmb3Igc3RhcnQsIGVuZCBpbiB6aXAoc2VnbWVudHNbOi0xXSwgc2VnbWVudHNbMTpdLCBzdHJpY3Q9VHJ1ZSk6CiAgICAgICAgICAgICAgICAjIGF0dGVudGlvbl9tYXNrPU5vbmUgLT4gc2RwYS9lYWdlciB0YWtlIHRoZWlyIG93biBgaXNfY2F1c2FsYCBwYXRoLCB3aGljaCBpcwogICAgICAgICAgICAgICAgIyBleGFjdGx5IHJpZ2h0IGluc2lkZSBvbmUgcGFja2VkIHNhbXBsZSBhbmQga2VlcHMgdGhlIGtlcm5lbCBPKG4pIGluIG1lbW9yeQogICAgICAgICAgICAgICAgb3V0LCBfID0gYXR0ZW50aW9uX2ludGVyZmFjZSgKICAgICAgICAgICAgICAgICAgICBzZWxmLAogICAgICAgICAgICAgICAgICAgIHF1ZXJ5X3N0YXRlc1s6LCA6LCBzdGFydDplbmRdLAogICAgICAgICAgICAgICAgICAgIGtleV9zdGF0ZXNbOiwgOiwgc3RhcnQ6ZW5kXSwKICAgICAgICAgICAgICAgICAgICB2YWx1ZV9zdGF0ZXNbOiwgOiwgc3RhcnQ6ZW5kXSwKICAgICAgICAgICAgICAgICAgICBOb25lLAogICAgICAgICAgICAgICAgICAgICoqYXR0bl9jb21tb24sCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgICAgICBvdXRwdXRzLmFwcGVuZChvdXQpCiAgICAgICAgICAgIGF0dG5fb3V0cHV0ID0gb3V0cHV0c1swXSBpZiBsZW4ob3V0cHV0cykgPT0gMSBlbHNlIHRvcmNoLmNhdChvdXRwdXRzLCBkaW09MSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBhdHRuX291dHB1dCwgYXR0bl93ZWlnaHRzID0gYXR0ZW50aW9uX2ludGVyZmFjZSgKICAgICAgICAgICAgICAgIHNlbGYsCiAgICAgICAgICAgICAgICBxdWVyeV9zdGF0ZXMsCiAgICAgICAgICAgICAgICBrZXlfc3RhdGVzLAogICAgICAgICAgICAgICAgdmFsdWVfc3RhdGVzLAogICAgICAgICAgICAgICAgYXR0ZW50aW9uX21hc2ssCiAgICAgICAgICAgICAgICAqKmF0dG5fY29tbW9uLAogICAgICAgICAgICApCiAgICBmaW5hbGx5OgogICAgICAgIHNlbGYubnVtX2tleV92YWx1ZV9ncm91cHMgPSBvcmlnX2dyb3VwcwoKICAgIGlmIGRvX2FsbF90b19hbGw6CiAgICAgICAgIyAoYnN6LCBzZXFfbGVuLCBuX2hlYWQvc3AsIGhlYWRfZGltKSAtPiAoYnN6LCBzZXFfbGVuL3NwLCBuX2hlYWQsIGhlYWRfZGltKQogICAgICAgIGF0dG5fb3V0cHV0ID0gZ2F0aGVyX2hlYWRzX3NjYXR0ZXJfc2VxKGF0dG5fb3V0cHV0LCBzZXFfZGltPTEsIGhlYWRfZGltPTIpCgogICAgYXR0bl9vdXRwdXQgPSBhdHRuX291dHB1dC5yZXNoYXBlKCppbnB1dF9zaGFwZSwgLTEpLmNvbnRpZ3VvdXMoKQogICAgYXR0bl9vdXRwdXQgPSBhdHRuX291dHB1dCAqIHRvcmNoLnNpZ21vaWQoZ2F0ZSkKICAgIHJldHVybiBzZWxmLm9fcHJvaihhdHRuX291dHB1dCksIGF0dG5fd2VpZ2h0cw=="
FN_SRC = base64.b64decode(FN_B64).decode()

ANCHOR_FN = "def qwen3_5_decoder_layer_forward("

# the decoder layer pops cu_seqlens out of kwargs; the attention wrapper needs it back
DL_OLD_B64 = "ICAgIGVsaWYgZ2V0YXR0cihzZWxmLCAibGF5ZXJfdHlwZSIsIGdldGF0dHIoc2VsZiwgImJsb2NrX3R5cGUiLCBOb25lKSkgPT0gImZ1bGxfYXR0ZW50aW9uIjoKICAgICAgICBoaWRkZW5fc3RhdGVzLCBfID0gc2VsZi5zZWxmX2F0dG4oCiAgICAgICAgICAgIGhpZGRlbl9zdGF0ZXM9aGlkZGVuX3N0YXRlcywKICAgICAgICAgICAgYXR0ZW50aW9uX21hc2s9YXR0ZW50aW9uX21hc2ssCiAgICAgICAgICAgIHBvc2l0aW9uX2lkcz1wb3NpdGlvbl9pZHMsCiAgICAgICAgICAgIHBhc3Rfa2V5X3ZhbHVlcz1wYXN0X2tleV92YWx1ZXMsCiAgICAgICAgICAgIHBvc2l0aW9uX2VtYmVkZGluZ3M9cG9zaXRpb25fZW1iZWRkaW5ncywKICAgICAgICAgICAgKiprd2FyZ3MsCiAgICAgICAgKQ=="
DL_NEW_B64 = "ICAgIGVsaWYgZ2V0YXR0cihzZWxmLCAibGF5ZXJfdHlwZSIsIGdldGF0dHIoc2VsZiwgImJsb2NrX3R5cGUiLCBOb25lKSkgPT0gImZ1bGxfYXR0ZW50aW9uIjoKICAgICAgICBhdHRuX2t3YXJncyA9IGt3YXJncwogICAgICAgIGlmIGdldF91bHlzc2VzX3NlcXVlbmNlX3BhcmFsbGVsX3dvcmxkX3NpemUoKSA+IDE6CiAgICAgICAgICAgICMgYHF3ZW4zXzVfYXR0bl9mb3J3YXJkYCBuZWVkcyB0aGUgZ2xvYmFsICh1bnNsaWNlZCkgcGFja2luZyBib3VuZGFyaWVzIHRvIGtlZXAKICAgICAgICAgICAgIyBhdHRlbnRpb24gYmxvY2stZGlhZ29uYWwgYWZ0ZXIgdGhlIGFsbC10by1hbGwgLS0gSEYncyBvd24gbWFzayBpcyBidWlsdCBmcm9tIHRoZQogICAgICAgICAgICAjIGxvY2FsIHNoYXJkIGFuZCBpcyBtZWFuaW5nbGVzcyB0aGVyZS4gT25seSBjb25zdW1lZCB3aGVuIHRoYXQgcGF0Y2ggaXMgaW5zdGFsbGVkLgogICAgICAgICAgICBhdHRuX2t3YXJncyA9IHsqKmt3YXJncywgImN1X3NlcWxlbnMiOiBjdV9zZXFsZW5zLCAiY3Vfc2VxbGVuc19jcHUiOiBjdV9zZXFsZW5zX2NwdX0KICAgICAgICBoaWRkZW5fc3RhdGVzLCBfID0gc2VsZi5zZWxmX2F0dG4oCiAgICAgICAgICAgIGhpZGRlbl9zdGF0ZXM9aGlkZGVuX3N0YXRlcywKICAgICAgICAgICAgYXR0ZW50aW9uX21hc2s9YXR0ZW50aW9uX21hc2ssCiAgICAgICAgICAgIHBvc2l0aW9uX2lkcz1wb3NpdGlvbl9pZHMsCiAgICAgICAgICAgIHBhc3Rfa2V5X3ZhbHVlcz1wYXN0X2tleV92YWx1ZXMsCiAgICAgICAgICAgIHBvc2l0aW9uX2VtYmVkZGluZ3M9cG9zaXRpb25fZW1iZWRkaW5ncywKICAgICAgICAgICAgKiphdHRuX2t3YXJncywKICAgICAgICAp"
DL_OLD = base64.b64decode(DL_OLD_B64).decode()
DL_NEW = base64.b64decode(DL_NEW_B64).decode()

IMPORT_OLD = """        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,"""
IMPORT_NEW = """        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Attention,
            Qwen3_5DecoderLayer,"""

IMPORT_MOE_OLD = """        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,"""
IMPORT_MOE_NEW = """        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention,
            Qwen3_5MoeDecoderLayer,"""

IMPORT_VERL_OLD = """        from verl.models.transformers.qwen3_5 import (
            fast_pos_embed_interpolate,
            forward_with_normal_backend,
            qwen3_5_base_forward,"""
IMPORT_VERL_NEW = """        from verl.models.transformers.qwen3_5 import (
            fast_pos_embed_interpolate,
            forward_with_normal_backend,
            qwen3_5_attn_forward,
            qwen3_5_base_forward,"""

REG_OLD = """        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen3_5TextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen3_5MoeTextModel)
"""
REG_NEW = """        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(Qwen3_5TextModel)
            patch_vlm_for_ulysses_input_slicing(Qwen3_5MoeTextModel)

            # Step 3: full-attention layers need the all-to-all for every backend, not just
            # flash attention. The generic `_flash_attention_forward` hook below is never
            # reached under e.g. `attn_implementation=sdpa`, which would silently confine
            # attention to each rank's own 1/sp slice of the sequence.
            Qwen3_5Attention.forward = qwen3_5_attn_forward
            Qwen3_5MoeAttention.forward = qwen3_5_attn_forward
            print(f"Monkey patch {model.__class__.__name__} attention layer for Ulysses SP")
"""


def verl_dir():
    spec = importlib.util.find_spec("verl")
    return os.path.dirname(spec.origin)


def patch_model_file(path, check):
    src = open(path).read()
    if "def qwen3_5_attn_forward(" in src and DL_NEW in src:
        return "already patched"
    if check:
        return "MISSING qwen3_5_attn_forward"
    assert ANCHOR_FN in src, f"anchor not found in {path}"
    assert src.count(DL_OLD) == 1, f"expected 1 self_attn call site in {path}, got {src.count(DL_OLD)}"
    open(path + ".orig", "w").write(src)
    # FN_SRC is rstripped; PEP8 wants two blank lines before the anchor def
    new = src.replace(ANCHOR_FN, FN_SRC + "\n\n\n" + ANCHOR_FN, 1).replace(DL_OLD, DL_NEW, 1)
    open(path, "w").write(new)
    return "patched"


def patch_monkey_file(path, check):
    src = open(path).read()
    done = "Qwen3_5Attention.forward = qwen3_5_attn_forward" in src
    if done:
        return "already patched"
    if check:
        return "MISSING attention registration"
    new = src
    for old, rep in (
        (IMPORT_OLD, IMPORT_NEW),
        (IMPORT_MOE_OLD, IMPORT_MOE_NEW),
        (IMPORT_VERL_OLD, IMPORT_VERL_NEW),
        (REG_OLD, REG_NEW),
    ):
        assert new.count(old) == 1, f"expected exactly 1 occurrence of:\n{old}\ngot {new.count(old)}"
        new = new.replace(old, rep, 1)
    open(path + ".orig", "w").write(src)
    open(path, "w").write(new)
    return "patched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = verl_dir()
    m = os.path.join(d, "models", "transformers", "qwen3_5.py")
    p = os.path.join(d, "models", "transformers", "monkey_patch.py")
    r1 = patch_model_file(m, a.check)
    r2 = patch_monkey_file(p, a.check)
    print(f"{m}: {r1}")
    print(f"{p}: {r2}")
    if a.check:
        ok = r1 == "already patched" and r2 == "already patched"
        # runtime proof: the class attribute really gets swapped
        import torch  # noqa: F401
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Attention,
            Qwen3_5TextConfig,
            Qwen3_5TextModel,
        )

        from verl.models.transformers.monkey_patch import apply_monkey_patch

        # heads must be divisible by ulysses_sp_size (apply_monkey_patch asserts it)
        cfg = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=8, num_key_value_heads=8, head_dim=16,
            layer_types=["full_attention"] * 2, max_position_embeddings=32,
        )
        cfg._attn_implementation = "sdpa"
        mdl = Qwen3_5TextModel(cfg)
        cfg.model_type = "qwen3_5"
        apply_monkey_patch(mdl, ulysses_sp_size=8, use_remove_padding=True, use_fused_kernels=False)
        name = Qwen3_5Attention.forward.__name__
        print(f"Qwen3_5Attention.forward -> {name}")
        ok = ok and name == "qwen3_5_attn_forward"
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
