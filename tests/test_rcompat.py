import numpy as np
import pytest

from spidepy._rcompat import p_adjust_bh, r_num_str, r_sort_str, sanitise


def test_r_num_str_matches_r_paste0(ref):
    d = ref("rcompat")
    vals, expected = np.atleast_1d(d["num"]), np.atleast_1d(d["num_str"])
    got = [r_num_str(v) for v in vals]
    assert got == list(expected)


@pytest.mark.parametrize(
    "x, s",
    [(10, "10"), (30, "30"), (2.5, "2.5"), (0.1, "0.1"), (1e5, "1e+05"), (123456789, "123456789"),
     (1e-5, "1e-05"), (1 / 3, "0.333333333333333"), (100000, "1e+05"), (1e15, "1e+15")],
)  # fmt: skip
def test_r_num_str_brief_values(x, s):
    assert r_num_str(x) == s


def test_r_num_str_missing():
    assert r_num_str(None) == "NA"


def test_p_adjust_bh_matches_r_with_na(ref):
    d = ref("rcompat")
    got = p_adjust_bh(d["p"])
    exp = np.asarray(d["p_bh"])
    np.testing.assert_array_equal(np.isnan(got), np.isnan(exp))
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-12)


def test_p_adjust_bh_no_na_and_all_na(ref):
    d = ref("rcompat")
    np.testing.assert_allclose(p_adjust_bh(d["p_all"]), d["p_all_bh"], rtol=1e-10, atol=1e-12)
    assert np.isnan(p_adjust_bh([np.nan, np.nan])).all()
    assert p_adjust_bh([]).shape == (0,)


def test_sanitise_matches_r(ref):
    d = ref("rcompat")
    assert sanitise(list(d["san_in"])) == list(d["san_out"])
    assert sanitise("Non-responder") == "Non.responder"


def test_r_sort_str_is_c_collation(ref):
    d = ref("rcompat")
    # R's method = "radix" is locale independent and equals code-point order
    assert r_sort_str(d["strs"]) == list(d["sort_radix"])
    assert r_sort_str(["S1", "S10", "S2"]) == ["S1", "S10", "S2"]
    # plain sort() agrees whenever the session locale is C
    if str(d["locale_collate"]) == "C":
        assert r_sort_str(d["strs"]) == list(d["sort_default"])
