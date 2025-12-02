# Ideas on how to do contraction with bayesian quimb model, in a way to consider samples and batches.

here in this code they seem to handle the model differently. Input are not nodes, and the nodes are exclusively the parameters of the model. then define contractions as such

```py

    def predict(self, sample: np.ndarray, embedding: Embedding = TrigonometricEmbedding(), return_tn: bool = False, normalize: bool = False) -> Union[np.ndarray, qtn.TensorNetwork]:
        """ Predicts the output of the model.
        
        Parameters
        ----------
        sample : :class:`numpy.ndarray`
            Input data.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        return_tn : bool   
            If True, returns tensor network, otherwise returns data. Useful when you want to vmap over predict function.
        
        Returns
        -------
        :class:`quimb.tensor.tensor_core.TensorNetwork`
            Output of the model.
        """

        if len(sample.flatten()) < self.L:
            raise ValueError(f"Input data must have at least {self.L} elements!")

        tn_sample = embed(sample, embedding)
        
        if callable(getattr(self, "apply", None)):
            output = self.apply(tn_sample)
        else:
            output = self & tn_sample
        
        if return_tn:
            return output
        else:
            output = output.contract(all, optimize='auto-hq')
            if type(output) == qtn.Tensor:
                y_pred = output.squeeze().data
            else:
                y_pred = output.squeeze()
            if normalize:
                y_pred = y_pred/jnp.linalg.norm(y_pred)
            return y_pred
        
    def forward(self, data: jnp.ndarray, embedding: Embedding = TrigonometricEmbedding(), batch_size: int=64, normalize: bool = False, dtype: Any = jnp.float_, seed: int = 42, alternate_flip: bool = False) -> Collection:
        """ Forward pass of the model.

        Parameters
        ----------
        data : :class:`jax.numpy.ndarray`
            Input data.
        y_true: :class:`jax.numpy.ndarray`
            Target class vector.
        embedding: :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        batch_size: int
            Batch size for data processing.
        normalize: bool
            If True, the model output is normalized.
        dtype: Any
            Data type of input data.
        seed: int
            Random seed for data shuffling.
        
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Output of the model.
        """
        outputs = []
        for batch_data in _batch_iterator(data, batch_size=batch_size, shuffle=False, dtype=dtype, seed=seed, alternate_flip=alternate_flip):
            x = jnp.array(batch_data, dtype=jnp.float64)
            x = jax.device_put(x, device=jax.devices(self.device[0])[self.device[1]])

            output = jnp.squeeze(jnp.array(jax.vmap(self.predict, in_axes=(0, None, None, None))(x, embedding, False, normalize)))
            outputs.append(output)
        
        return jnp.concatenate(outputs, axis=0)
```

for further references the repository is

https://github.com/bsc-quantic/tn4ml

and this file at

https://github.com/bsc-quantic/tn4ml/blob/2fbd658005449777bcd037a08fd863d7bd948475/tn4ml/models/model.py#L256


# Generalization.

Of course we would want to not be constrained by the structure, and so in the model there should be a set of label, input_labels that indicates the mode contracting to the inputs. so maybe the class should have an object of the input that through label can do the forward. of course for repeated inputs we dont need to copy the same tensor.

This in order to have an output of the forward of dim (samples, out_dim)
