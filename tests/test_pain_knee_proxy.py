import unittest

import torch
from phc.env.util.pain_baseline import (
    combine_knee_oa_load_proxy,
    compute_knee_contact_load_proxy,
    compute_knee_moment_load_proxy,
    compute_knee_torque_load_proxy,
)


class KneeProxyTest(unittest.TestCase):
    def test_knee_torque_proxy_uses_explicit_reference_not_asset_effort_limit(self):
        tau = torch.tensor([50.0])
        dq = torch.tensor([0.0])
        rom_proxy = torch.tensor([0.0])
        huge_asset_effort_limit = torch.tensor([1.0e39])

        load, components = compute_knee_torque_load_proxy(
            tau=tau,
            dq=dq,
            rom_proxy=rom_proxy,
            asset_tau_limit=huge_asset_effort_limit,
            torque_ref=100.0,
            power_ref=5.0,
            w_torque=1.0,
            w_flex=0.0,
            w_rom=0.0,
            w_work=0.0,
        )

        self.assertTrue(torch.allclose(components["torque"], torch.tensor([0.5])))
        self.assertTrue(torch.allclose(load, torch.tensor([0.5])))

    def test_knee_load_still_reports_raw_tau_and_work_components(self):
        tau = torch.tensor([10.0])
        dq = torch.tensor([2.0])
        rom_proxy = torch.tensor([0.25])

        load, components = compute_knee_torque_load_proxy(
            tau=tau,
            dq=dq,
            rom_proxy=rom_proxy,
            asset_tau_limit=torch.tensor([1000.0]),
            torque_ref=100.0,
            power_ref=5.0,
            w_torque=0.5,
            w_flex=0.25,
            w_rom=0.25,
            w_work=0.1,
        )

        self.assertTrue(torch.allclose(components["tau_abs"], torch.tensor([10.0])))
        self.assertTrue(torch.allclose(components["work"], torch.tensor([4.0])))
        self.assertTrue(torch.allclose(load, torch.tensor([0.5375])))

    def test_knee_contact_proxy_uses_stance_foot_load_not_actuator_torque(self):
        foot_force = torch.tensor([[0.0, 0.0, 500.0], [30.0, 40.0, 0.0]])
        knee_pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
        foot_pos = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        knee_flex = torch.tensor([0.4, 0.4])

        load, components = compute_knee_contact_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            knee_flex=knee_flex,
            body_weight_ref=500.0,
            flex_compression_gain=0.5,
            loading_rate=None,
            loading_rate_ref=1000.0,
            w_compression=1.0,
            w_loaded_flex=0.0,
            w_loading_rate=0.0,
        )

        self.assertTrue(torch.allclose(components["compression"], torch.tensor([1.0, 0.1])))
        self.assertTrue(torch.allclose(load, torch.tensor([1.0, 0.1])))

    def test_knee_moment_proxy_separates_kam_and_kfm_geometry(self):
        knee_pos = torch.tensor([[0.0, 0.0, 0.5]])
        foot_pos = torch.tensor([[0.1, -0.2, 0.0]])
        foot_force = torch.tensor([[0.0, 0.0, 500.0]])

        load, components = compute_knee_moment_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            kam_ref=50.0,
            kfm_ref=50.0,
            w_kam=1.0,
            w_kfm=0.5,
        )

        self.assertTrue(torch.allclose(components["kam"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(components["kfm"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(load, torch.tensor([2.5])))

    def test_oa_load_combiner_keeps_components_visible(self):
        contact = torch.tensor([0.5])
        moment = torch.tensor([0.25])
        torque = torch.tensor([0.1])

        load, components = combine_knee_oa_load_proxy(
            contact_load=contact,
            moment_load=moment,
            torque_load=torque,
            w_contact=0.7,
            w_moment=0.3,
            w_torque=0.0,
        )

        self.assertTrue(torch.allclose(load, torch.tensor([0.425])))
        self.assertTrue(torch.allclose(components["contact_load"], contact))
        self.assertTrue(torch.allclose(components["moment_load"], moment))
        self.assertTrue(torch.allclose(components["torque_load"], torque))


if __name__ == "__main__":
    unittest.main()
