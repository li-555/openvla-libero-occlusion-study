# Gripper audit version

This version preserves the existing contact gate and adds only a read-only
`gripper_audit.py` diagnostic. `config.xml_reference.json` uses the Panda XML
joint-limit endpoints (`q0-q1 = 0` closed and `0.08` open) as an engineering
reference sensitivity run. It is **not** an empirical calibration file and
must not be used as evidence of contact classification accuracy.

The original source trace and prior result directory are unchanged.
