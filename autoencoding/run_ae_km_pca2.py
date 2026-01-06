
    # PCA(2) of latent space + explained variance ratio.
    pca2 = PCA(n_components=2, random_state=0)
    Z2 = pca2.fit_transform(Z_elbow).astype(np.float32, copy=False)
    evr = pca2.explained_variance_ratio_.astype(np.float32, copy=False)

    np.savez_compressed(
        out_root / "ae_latent_pca2.npz",
        Z2=Z2,
        explained_variance_ratio=evr,
    )