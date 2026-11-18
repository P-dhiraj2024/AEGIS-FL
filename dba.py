"""Distributed backdoor attack (Xie et al., ICLR 2020): malicious client i stamps local trigger
part i mod 4 on half of its batches and relabels them to the target class; the test-time
trigger is the union of the 4 parts (data.stamp_trigger).  Training: fl/client.Poison."""
