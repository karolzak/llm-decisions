"""Black-box probing of decision APIs ("System One" models) and their open-weight analogues.

`api` defines vendor-neutral request/response types, `clients` adapts them to each vendor's
wire format, `probes` defines the experiments, `runner` executes and stores them, and
`forensics` / `report` analyse the stored responses.
"""
