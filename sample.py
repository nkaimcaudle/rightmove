import hashlib
import inspect
import json
from datetime import datetime
from secrets import token_urlsafe
import sys

import arviz as az
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
import pandas as pd
import json_numpy
import hsgp_nD
import utils_data

numpyro.set_host_device_count(4)
numpyro.enable_x64(True)

sys.path.append("/home/nkaimcaudle/Projects/numpyro/")


def get_data() -> dict:
    df = utils_data.get_property_data()

    tags = df["propertyData.tags"].apply(pd.Series, dtype="object")

    px = df["analyticsInfo.analyticsProperty.price"].apply(np.log).values - np.log(1e6)
    beds = df["analyticsInfo.analyticsProperty.beds"].values
    bathrooms = df["propertyData.bathrooms"].values
    ksqft = df["analyticsInfo.analyticsProperty.maxSizeFt"].divide(1e3).values
    postcode_idx, postcodes = df["postcode_head"].factorize(sort=True)
    subtype_idx, subtypes = df[
        "analyticsInfo.analyticsProperty.propertySubType"
    ].factorize(sort=True)
    tenure_idx, tenures = df["propertyData.tenure.tenureType"].factorize(sort=True)
    is_new_home = tags.eq("NEW_HOME").any(1).astype(int).values
    G = df.loc[
        :,
        [
            "analyticsInfo.analyticsProperty.longitude",
            "analyticsInfo.analyticsProperty.latitude",
        ],
    ].values
    G = (G - G.min(0, keepdims=True)) / (
        G.max(0, keepdims=True) - G.min(0, keepdims=True)
    )
    G = G * 2 - 1.0

    dims = {
        "beta_postcode": ["postcodes"],
        "beta_subtype": ["subtypes"],
        "beta_tenure": ["tenures"],
    }
    coords = {"postcodes": postcodes, "subtypes": subtypes, "tenures": tenures}
    data = {
        "beds": beds,
        "bathrooms": bathrooms,
        "postcode_idx": postcode_idx,
        "postcode_K": len(postcodes),
        "subtype_idx": subtype_idx,
        "subtype_K": len(subtypes),
        "tenure_idx": tenure_idx,
        "tenure_K": len(tenures),
        "ksqft": ksqft,
        "is_new_home": is_new_home,
        "G": G,
    }
    dct = {"dims": dims, "coords": coords, "observed": {"px": px}, "data": data}
    return dct


def model_orig(
    beds,
    bathrooms,
    postcode_idx,
    postcode_K,
    subtype_idx,
    subtype_K,
    tenure_idx,
    tenure_K,
    ksqft,
    is_new_home,
    G,
    px=None,
):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    df = numpyro.sample("df", dist.Gamma(2.5, 0.1))
    scale = numpyro.sample("scale", dist.Exponential(1.0))

    beta_beds = numpyro.sample("beta_beds", dist.Normal(0, 0.2))
    beta_baths = numpyro.sample("beta_baths", dist.Normal(0, 0.2))
    beta_is_new_home = numpyro.sample("beta_is_new_home", dist.Normal(0, 0.2))

    with numpyro.plate("plate_postcode", postcode_K):
        beta_postcode = numpyro.sample("beta_postcode", dist.Normal(0, 0.2))
    with numpyro.plate("plate_subtype", subtype_K):
        beta_subtype = numpyro.sample("beta_subtype", dist.Normal(0, 0.2))
    with numpyro.plate("plate_tenure", tenure_K - 1):
        beta_tenure_raw = numpyro.sample("beta_tenure_raw", dist.Normal(0, 0.2))
    beta_tenure = numpyro.deterministic(
        "beta_tenure", jnp.r_[beta_tenure_raw[0], 0.0, beta_tenure_raw[-1]]
    )

    # ksqft_alpha = numpyro.sample("ksqft_alpha", dist.HalfCauchy(1.0))
    # ksqft_beta = numpyro.sample("ksqft_beta", dist.HalfCauchy(1.0))
    # with numpyro.plate("plate_ksqft_impute", np.isnan(ksqft).sum()):
    #     ksqft_impute = numpyro.sample(
    #         "ksqft_impute",
    #         dist.Gamma(ksqft_alpha, ksqft_beta).mask(False),
    #     )
    # ksqft_merge = jnp.asarray(ksqft).at[np.nonzero(np.isnan(ksqft))[0]].set(ksqft_impute)
    # ksqft_obs = numpyro.sample(
    #     "ksqft_obs", dist.Gamma(ksqft_alpha, ksqft_beta), obs=ksqft_merge
    # )
    # beta_ksqft = numpyro.sample("beta_ksqft", dist.Normal(0, 0.2))

    loc = a + beta_beds * (beds - 2.0)
    loc += beta_baths * (bathrooms - 1.0)
    loc += beta_postcode[postcode_idx]
    loc += beta_subtype[subtype_idx]
    loc += beta_tenure[tenure_idx]
    # loc += beta_ksqft * ksqft
    loc += beta_is_new_home * is_new_home

    obs = numpyro.sample("obs", dist.StudentT(df, loc, scale), obs=px)


def model_2d(
    beds,
    bathrooms,
    postcode_idx,
    postcode_K,
    subtype_idx,
    subtype_K,
    tenure_idx,
    tenure_K,
    ksqft,
    is_new_home,
    G,
    M,
    L,
    px=None,
):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    df = numpyro.sample("df", dist.Gamma(2.5, 0.1))
    scale = numpyro.sample("scale", dist.Exponential(1.0))

    beta_beds = numpyro.sample("beta_beds", dist.Normal(0, 0.2))
    beta_baths = numpyro.sample("beta_baths", dist.Normal(0, 0.2))
    beta_is_new_home = numpyro.sample("beta_is_new_home", dist.Normal(0, 0.2))

    # with numpyro.plate("plate_postcode", postcode_K):
    #     beta_postcode = numpyro.sample("beta_postcode", dist.Normal(0, 0.2))
    with numpyro.plate("plate_subtype", subtype_K):
        beta_subtype = numpyro.sample("beta_subtype", dist.Normal(0, 0.2))
    with numpyro.plate("plate_tenure", tenure_K - 1):
        beta_tenure_raw = numpyro.sample("beta_tenure_raw", dist.Normal(0, 0.2))
    beta_tenure = numpyro.deterministic(
        "beta_tenure", jnp.r_[beta_tenure_raw[0], 0.0, beta_tenure_raw[-1]]
    )

    N, D = G.shape
    M_nD = M**D
    indices = hsgp_nD.get_indices(M, D)
    phi = hsgp_nD.get_phi(N, M_nD, L, indices, G)

    alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
    rho = numpyro.sample("rho", dist.HalfNormal(2.0))
    with numpyro.plate("plate_", M_nD):
        beta = numpyro.sample("beta", dist.Normal(0, 1))

    diagSPD = jnp.empty(M_nD)
    for m in range(M_nD):
        lam = hsgp_nD.lambda_nD(
            L,
            indices[
                m,
            ],
            D,
        )
        diagSPD = diagSPD.at[m].set(
            jnp.sqrt(hsgp_nD.spd_nD_onelscale(alpha, rho, jnp.sqrt(lam), D))
        )
    SPD_beta = diagSPD * beta
    f = numpyro.deterministic("f", phi.dot(SPD_beta))

    loc = a + beta_beds * (beds - 2.0)
    loc += beta_baths * (bathrooms - 1.0)
    # loc += beta_postcode[postcode_idx]
    loc += f
    loc += beta_subtype[subtype_idx]
    loc += beta_tenure[tenure_idx]
    # loc += beta_ksqft * ksqft
    loc += beta_is_new_home * is_new_home

    obs = numpyro.sample("obs", dist.StudentT(df, loc, scale), obs=px)


def model_just_2d(G, M, L):
    N, D = G.shape
    M_nD = M**D
    indices = hsgp_nD.get_indices(M, D)
    phi = hsgp_nD.get_phi(N, M_nD, L, indices, G)

    alpha = numpyro.sample("alpha", dist.HalfNormal(2.0))
    rho = numpyro.sample("rho", dist.HalfNormal(2.0))
    with numpyro.plate("plate_", M_nD):
        beta = numpyro.sample("beta", dist.Normal(0, 1))

    diagSPD = jnp.empty(M_nD)
    for m in range(M_nD):
        lam = hsgp_nD.lambda_nD(
            L,
            indices[
                m,
            ],
            D,
        )
        diagSPD = diagSPD.at[m].set(
            jnp.sqrt(hsgp_nD.spd_nD_onelscale(alpha, rho, jnp.sqrt(lam), D))
        )
    SPD_beta = diagSPD * beta
    f = numpyro.deterministic("f", phi.dot(SPD_beta))
    return f


def run_model(model, dims, coords, observed, return_hmc=False, **kwargs):
    rng_prior, rng_sample, rng_post = random.split(random.PRNGKey(32), 3)
    prior = Predictive(model, num_samples=500)(rng_prior, **kwargs)
    hmc = MCMC(
        NUTS(model, init_strategy=init_to_median),
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        progress_bar=True,
    )
    hmc.run(rng_sample, extra_fields=("z", "energy", "diverging"), **observed, **kwargs)
    ppc = Predictive(model, hmc.get_samples())(rng_post, **kwargs)
    idata = az.from_numpyro(
        hmc, coords=coords, dims=dims, posterior_predictive=ppc, prior=prior
    )
    if return_hmc:
        return idata, hmc
    else:
        return idata


def get_small_varnames(idata):
    small = []
    for var_name in idata.posterior:
        shape = idata.posterior[var_name].shape
        # print(var_name, shape, len(shape))
        if len(shape) == 2:
            small.append(var_name)
        elif np.product(shape[2:]) <= 4:
            small.append(var_name)
        else:
            pass
    return small


def main() -> int:
    dct = get_data()
    dims = dct.pop("dims")
    coords = dct.pop("coords")
    observed = dct.pop("observed")
    data = dct.pop("data")
    data["M"] = 18
    data["L"] = 1.5

    model = model_2d

    idata = run_model(model, dims, coords, observed=observed, **data)
    data_hash = utils_data.get_dict_of_np_hash(data)
    model_hash = hashlib.sha256(inspect.getsource(model).encode()).hexdigest()
    token = token_urlsafe(4).replace("-", "_")
    results = {"model_hash": model_hash, "data_hash": data_hash}
    results["model_source"] = inspect.getsource(model)
    results["utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"models/{token}.args.json", "w") as f:
        json.dump(results, f)
    with open(f"models/{token}.data.json", "w") as f:
        json_numpy.dump(data, f)
    idata.to_netcdf(f"models/{token}.cdf")
    print(f"saved models/{token}.cdf")
    print(f"saved models/{token}.args.json")
    print(f"saved models/{token}.data.json")

    return 0


if __name__ == "__main__":
    exit(main())
