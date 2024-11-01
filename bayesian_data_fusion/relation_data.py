from concurrent.futures import ProcessPoolExecutor
from math import isnan

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, find
from sklearn.model_selection import train_test_split


class EntityModel:
    def __init__(
        self,
        sample=None,
        mu=None,
        Lambda=None,
        beta=None,
        mu0=None,
        b0=2.0,
        WI=None,
        nu0=0,
        uhat=None,
    ):
        """Initializes the EntityModel with specified latent vectors, parameters, and
        hyper-priors.

        Parameters:
        sample: np.array, matrix of latent vectors (default: zeros)
        mu: np.array, mean vector (default: zeros)
        Lambda: np.array, precision matrix (default: 5 * identity matrix)
        beta: np.array, matrix linking features to latent vectors (default: zeros)
        mu0: np.array, hyper-prior mean for Normal-Wishart (default: zeros)
        b0: float, scalar hyper-prior for Normal-Wishart
        WI: np.array, inverse of W (hyper-prior matrix for Normal-Wishart,
        default: identity matrix)
        nu0: int, degrees of freedom for Normal-Wishart
        uhat: np.array, temporary variable for uhat (default: zeros)
        """
        self.sample = sample if sample is not None else np.zeros((nu0, 0))
        self.mu = mu if mu is not None else np.zeros(nu0)
        self.Lambda = Lambda if Lambda is not None else 5 * np.eye(nu0)
        self.beta = beta if beta is not None else np.zeros((0, nu0))
        self.mu0 = mu0 if mu0 is not None else np.zeros(nu0)
        self.b0 = b0
        self.WI = WI if WI is not None else np.eye(nu0)
        self.nu0 = nu0
        self.uhat = uhat if uhat is not None else np.zeros((0, 0))


class Entity:
    def __init__(
        self,
        F=None,
        FF=None,
        use_FF=False,
        Frefs=None,
        relations=None,
        count=0,
        name="",
        modes=None,
        modes_other=None,
        lambda_beta=1.0,
        lambda_beta_sample=True,
        mu=1.0,
        nu=1e-3,
        model=None,
    ):
        """Initializes an Entity with specified attributes for features, relations, and
        model parameters.

        Parameters:
        F: np.array, Feature matrix
        FF: np.array or None, Precomputed product F' * F
        use_FF: bool, Flag to indicate if FF should be used
        Frefs: list of Future, References to futures for parallel execution
        relations: list, Relations associated with this entity
        count: int, Number of instances associated with this entity
        name: str, Name of the entity
        modes: list of int, Modes for the entity
        modes_other: list of list of int, Alternative modes for the entity
        lambda_beta: float, Regularization parameter for beta
        lambda_beta_sample: bool, Indicator if lambda_beta is to be sampled
        mu: float, Hyper-prior for lambda_beta
        nu: float, Hyper-prior for lambda_beta
        model: EntityModel or None, Model associated with this entity
        """
        self.F = F if F is not None else np.zeros((0, 0))
        self.FF = FF
        self.use_FF = use_FF
        self.Frefs = Frefs if Frefs is not None else []
        self.relations = relations if relations is not None else []
        self.count = count
        self.name = name
        self.modes = modes if modes is not None else []
        self.modes_other = modes_other if modes_other is not None else []
        self.lambda_beta = lambda_beta
        self.lambda_beta_sample = lambda_beta_sample
        self.mu = mu
        self.nu = nu
        self.model = model

    @classmethod
    def with_name(cls, name, F=None, lambda_beta=1.0):
        """Alternate constructor to initialize an Entity with only name and optional F
        and lambda_beta.

        Parameters:
        name: str, Name of the entity
        F: np.array, Feature matrix (default: empty array)
        lambda_beta: float, Regularization parameter (default: 1.0)
        """
        return cls(
            F=F if F is not None else np.zeros((0, 0)),
            relations=[],
            count=0,
            name=name,
            lambda_beta=lambda_beta,
        )


def has_features(entity):
    """Checks if the entity has associated features."""
    return entity.F.size > 0


def init_model(
    entity,
    num_latent,
    sample=None,
    mu=None,
    Lambda=None,
    beta=None,
    mu0=None,
    b0=2.0,
    WI=None,
    nu0=None,
    uhat=None,
    lambda_beta=np.nan,
):
    """Initializes the model parameters for an Entity with customizable default
    arguments.

    Parameters:
    entity: Entity object to initialize the model for.
    num_latent: int, number of latent features.
    sample: np.array, latent sample matrix
    (default zeros with shape (num_latent, entity.count))
    mu: np.array, mean vector (default zeros with length num_latent)
    Lambda: np.array, precision matrix (default 5 * identity matrix of size num_latent)
    beta: np.array, linking matrix (default zeros if features are present, else empty array)
    mu0: np.array, hyper-prior mean for Normal-Wishart (default zeros with length num_latent)
    b0: float, scalar hyper-prior for Normal-Wishart
    WI: np.array, hyper-prior matrix for Normal-Wishart
    (default identity matrix of size num_latent)
    nu0: int, degrees of freedom for Normal-Wishart (default num_latent)
    uhat: np.array, temporary matrix for uhat
    (default zeros if features are present, else empty array)
    lambda_beta: float, optional lambda_beta value to set for the entity.
    """
    # Initialize the model instance and assign it to the entity
    m = EntityModel(
        sample=sample if sample is not None else np.zeros((num_latent, entity.count)),
        mu=mu if mu is not None else np.zeros(num_latent),
        Lambda=Lambda if Lambda is not None else 5 * np.eye(num_latent),
        beta=(
            beta
            if beta is not None
            else (
                np.zeros((entity.F.shape[1], num_latent))
                if has_features(entity)
                else np.zeros((0, num_latent))
            )
        ),
        mu0=mu0 if mu0 is not None else np.zeros(num_latent),
        b0=b0,
        WI=WI if WI is not None else np.eye(num_latent),
        nu0=nu0 if nu0 is not None else num_latent,
        uhat=(
            uhat
            if uhat is not None
            else (
                np.zeros((num_latent, entity.count)) if has_features(entity) else np.zeros((0, 0))
            )
        ),
    )
    entity.model = m

    # Update lambda_beta if provided
    if not isnan(lambda_beta):
        entity.lambda_beta = lambda_beta


def to_str(entity):
    """Generates a string representation of the entity, including its model's sample
    norm, beta norm (if features are present), and lambda_beta if it's sampled.

    Parameters:
    entity: Entity object whose information will be converted to a string.

    Returns:
    str: Formatted string representation of the entity.
    """
    if entity.model is None:
        return f"{entity.name[:3]}[]"

    # Start with entity name and sample norm
    result = f"{entity.name[:3]}[U:{np.linalg.norm(entity.model.sample):6.2f}"

    # Add beta norm if entity has features
    if has_features(entity):
        result += f" β:{np.linalg.norm(entity.model.beta):3.2f}"

    # Add lambda_beta if it is sampled
    if has_features(entity) and entity.lambda_beta_sample:
        result += f" λ={entity.lambda_beta:1.1f}"

    # Close the bracket and return
    result += "]"
    return result


class RelationModel:
    def __init__(
        self,
        alpha_sample=False,
        alpha_nu0=2.0,
        alpha_lambda0=1.0,
        lambda_beta=1.0,
        alpha=1.0,
        beta=None,
        mean_value=0.0,
    ):
        """Initializes a RelationModel with default parameters, allowing customization.

        Parameters:
        alpha_sample: bool, whether alpha is sampled
        alpha_nu0: float, hyperparameter for alpha sampling
        alpha_lambda0: float, hyperparameter for alpha sampling
        lambda_beta: float, regularization parameter for beta
        alpha: float, model regularization parameter
        beta: np.array, beta vector (default empty array)
        mean_value: float, mean value for the model
        """
        self.alpha_sample = alpha_sample
        self.alpha_nu0 = alpha_nu0
        self.alpha_lambda0 = alpha_lambda0
        self.lambda_beta = lambda_beta
        self.alpha = alpha
        self.beta = beta if beta is not None else np.zeros(0)
        self.mean_value = mean_value


class RelationTemp:
    def __init__(self, linear_values=None, FF=None):
        """Initializes a RelationTemp object to store temporary calculations.

        Parameters:
        linear_values: np.array, stores mean_value + F * beta (default: empty array)
        FF: np.array, stores the matrix product F' * F (default: empty matrix)
        """
        # Initialize linear_values and FF, defaulting to empty arrays if not provided
        self.linear_values = linear_values if linear_values is not None else np.zeros(0)
        self.FF = FF if FF is not None else np.zeros((0, 0))


class Relation:
    def __init__(
        self,
        data,
        name,
        entities=None,
        test_vec=None,
        test_F=None,
        test_label=None,
        class_cut=0.0,
        model=None,
        temp=None,
    ):
        """Initializes a Relation object to represent relationships between entities.

        Parameters:
        data: pd.DataFrame, Data representing relations.
        name: str, Name of the relation.
        entities: list of Entity, Entities involved in this relation.
        test_vec: pd.DataFrame, Test data for this relation.
        test_F: np.array, Feature matrix for the test data.
        test_label: np.array, Boolean labels for test data.
        class_cut: float, Cut-off value for classification.
        model: RelationModel, Model associated with the relation.
        temp: RelationTemp, Temporary storage for intermediate calculations.
        """
        self.data = data
        self.F = None  # Placeholder for features, to be set if needed
        self.entities = entities if entities is not None else []
        self.name = name
        self.test_vec = test_vec if test_vec is not None else pd.DataFrame()
        self.test_F = test_F
        self.test_label = test_label if test_label is not None else np.array([], dtype=bool)
        self.class_cut = class_cut
        self.model = model if model is not None else RelationModel(alpha=1.0)
        self.temp = temp if temp is not None else RelationTemp()

    @classmethod
    def with_data_and_entities(cls, data, name, entities=None, class_cut=0.0):
        """Alternate constructor for creating a Relation with predefined entities.

        Parameters:
        data: pd.DataFrame, Relation data between entities.
        name: str, Relation name.
        entities: list, Entities associated with this relation.
        class_cut: float, Classification cut-off.

        Returns:
        Relation: Instance of Relation with the provided data and entities.
        """
        dims = [data.iloc[:, i].max() for i in range(data.shape[1] - 1)]

        if entities:
            if len(entities) != data.shape[1] - 1:
                raise ValueError(
                    f"Data has {data.shape[1]} columns, but {len(entities) + 1} entities were expected."
                )

            for i, en in enumerate(entities):
                if en.count == 0:
                    en.count = dims[i]
                elif en.count > dims[i]:
                    dims[i] = en.count
                elif en.count < dims[i]:
                    raise ValueError(
                        f"Entity '{en.name}' has a smaller count ({en.count})"
                        f"than the largest id in the data ({dims[i]}). Set "
                        "entity.count manually before creating the relation."
                    )

        return cls(
            data=data,
            name=name,
            entities=entities,
            class_cut=class_cut,
            model=RelationModel(),
        )

    @classmethod
    def from_sparse_matrix(cls, data, name, entities=None, class_cut=0.0):
        """Alternate constructor to create a Relation from a sparse matrix.

        Parameters:
        data: csc_matrix, Sparse matrix representing relations.
        name: str, Name of the relation.
        entities: list, Entities associated with this relation
        (should contain exactly 2 entities).
        class_cut: float, Classification cut-off.

        Returns:
        Relation: Instance of Relation created from the sparse matrix.
        """
        if entities is None or len(entities) != 2:
            raise ValueError("For matrix relation, the number of entities must be 2.")

        # Extract non-zero indices and values from the sparse matrix
        U, V, X = find(data)

        # Create a DataFrame from non-zero entries
        df = pd.DataFrame({"E1": U, "E2": V, "values": X})

        return cls(data=df, name=name, entities=entities, class_cut=class_cut)

    def size(self, dimension=None):
        """Returns the size of the relation data.

        Parameters:
        dimension: int or None, specifies the dimension (1 or 2) for which size is required.

        Returns:
        tuple or int: Tuple with the sizes along each dimension if dimension is None,
                      otherwise the size along the specified dimension.
        """
        if dimension is None:
            return (self.data.shape[0], self.data.shape[1])
        elif dimension == 1:
            return self.data.shape[0]
        elif dimension == 2:
            return self.data.shape[1]
        else:
            raise ValueError("Dimension must be 1 or 2")

    def num_data(self):
        """Returns the number of non-zero entries in the relation data."""
        return np.count_nonzero(self.data)

    def num_test(self):
        """Returns the number of rows in the test vector data."""
        return self.test_vec.shape[0]

    def has_features(self):
        """Checks if the relation has features (if F is not empty)."""
        return self.F is not None and self.F.size > 0

    def to_str(self):
        """Returns a string representation of the relation, including model
        characteristics if present."""
        if self.model is None:
            return f"{self.name[:4]}[]"

        result = f"{self.name[:4]}[α={self.model.alpha:2.1f}"

        # Append beta norm if features are present
        if self.has_features():
            result += f" β:{np.linalg.norm(self.model.beta):2.1f}"

        result += "]"
        return result

    def assign_to_test(self, ntest):
        """Randomly assigns a subset of the relation data to the test set based on a
        specified sample size.

        Parameters:
        ntest: int, Number of samples to assign to the test set.
        """
        # Randomly select indices for test data
        test_id = np.random.choice(self.data.index, size=ntest, replace=False)
        self.assign_to_test_by_id(test_id)

    def assign_to_test_by_id(self, test_id):
        """Assigns specified rows of data to the test set based on the provided indices.

        Parameters:
        test_id: array-like, Row indices to assign to the test set.
        """
        self.test_vec = self.data.loc[test_id]
        self.data = self.data.drop(index=test_id)
        self.test_label = self.test_vec.iloc[:, -1].to_numpy() < self.class_cut

    def set_precision(self, precision):
        """Sets the precision value (alpha) in the model.

        Parameters:
        precision: float, New precision value to set.
        """
        self.model.alpha = precision

    def assign_to_test_by_id(self, test_id):
        """Assigns specified rows of data to the test set based on the provided indices.

        Parameters:
        test_id: list or np.array, Row indices to assign to the test set.
        """
        # Select rows based on test_id and update test_vec and data
        self.test_vec = self.data.loc[test_id]
        self.data = self.data.drop(index=test_id)

        # Set test labels based on the class cut-off
        self.test_label = (self.test_vec.iloc[:, -1] < self.class_cut).to_numpy()

        # Handle feature matrix F if available
        if self.has_features():
            self.test_F = self.F[test_id, :]
            train_mask = np.ones(self.F.shape[0], dtype=bool)
            train_mask[test_id] = False
            self.F = self.F[train_mask, :]

    def set_test(self, test_df, test_feat=None):
        """Assigns a given DataFrame to the test set with optional feature matrix.

        Parameters:
        test_df: pd.DataFrame, DataFrame containing test data.
        test_feat: np.array or None, Feature matrix associated with the test data.
        """
        # Validation checks for feature matrix dimensions
        if self.has_features() and test_feat is None:
            raise ValueError("Relation has features; please supply features with test data.")

        if self.has_features() and test_feat.shape[1] != self.F.shape[1]:
            raise ValueError("The test_feat must have the same number of columns as relation.F.")

        if self.has_features() and test_feat.shape[0] != test_df.shape[0]:
            raise ValueError("The test_feat must have the same number of rows as test_df.")

        if test_df.shape[1] != self.data.shape[1]:
            raise ValueError(
                "The number of columns in test_df must match the number"
                "of columns in relation.data."
            )

        # Assign test_df and set test labels
        self.test_vec = test_df
        self.test_label = (self.test_vec.iloc[:, -1] < self.class_cut).to_numpy()

        # Assign test_feat if features are present
        if self.has_features():
            self.test_F = test_feat

    def set_test_from_sparse(self, test_mat):
        """Sets the test data for the relation using a sparse matrix (CSC format).

        Parameters:
        test_mat: scipy.sparse.csc_matrix, Sparse matrix representing test relations.

        Raises:
        ValueError: If the relation has features or if the data format does not
        match expectations.
        """
        # Check if features are present, which prevents using sparse matrix for test data
        if self.has_features():
            raise ValueError(
                "Cannot add test set using SparseMatrixCSC when relation has features."
                "Use DataFrame instead."
            )

        # Check if the relation data has exactly 3 columns for two entities and values
        if self.data.shape[1] != 3:
            raise ValueError(
                "Relation must have 2 entities if using SparseMatrixCSC for test set."
            )

        # Extract non-zero entries from the sparse matrix
        row, col, data = find(test_mat)

        # Create a DataFrame for the test data
        test_df = pd.DataFrame(
            {
                self.data.columns[0]: row,
                self.data.columns[1]: col,
                self.data.columns[2]: data,
            }
        )

        # Assign to test_vec and set test labels based on class_cut
        self.test_vec = test_df
        self.test_label = (self.test_vec.iloc[:, -1] < self.class_cut).to_numpy()


class RelationData:
    def __init__(self, entities=None, relations=None):
        """Initializes a RelationData instance to store entities and relations.

        Parameters:
        entities: list of Entity, List of entities in the RelationData.
        relations: list of Relation, List of relations in the RelationData.
        """
        self.entities = entities if entities is not None else []
        self.relations = relations if relations is not None else []

    @classmethod
    def from_indexed_df(
        cls,
        Am,
        feat1=None,
        feat2=None,
        entity1="E1",
        entity2="E2",
        relation="Rel",
        ntest=0,
        class_cut=np.log10(200),
        alpha=5.0,
        alpha_sample=False,
        lambda_beta=1.0,
    ):
        """Alternate constructor to initialize RelationData with a set of parameters and
        indexed data.

        Parameters:
        Am: IndexedDF, Data structure holding indexed relational data.
        feat1: np.array, Feature matrix for the first entity.
        feat2: np.array, Feature matrix for the second entity.
        entity1: str, Name for the first entity.
        entity2: str, Name for the second entity.
        relation: str, Name for the relation.
        ntest: int, Number of test samples.
        class_cut: float, Classification cut-off.
        alpha: float, Alpha parameter for relation model.
        alpha_sample: bool, Whether to sample alpha in relation model.
        lambda_beta: float, Lambda parameter for beta.

        Returns:
        RelationData: An instance of RelationData.
        """
        # Create relation based on alpha_sample
        r = (
            Relation(Am, relation, class_cut=class_cut)
            if alpha_sample
            else Relation(Am, relation, class_cut=class_cut, alpha=alpha)
        )

        # Create entities and associate them with the relation
        e1 = Entity(
            F=feat1 if feat1 is not None else np.zeros((0, 0)),
            relations=[r],
            count=r.size(1),
            name=entity1,
            lambda_beta=lambda_beta,
        )

        e2 = Entity(
            F=feat2 if feat2 is not None else np.zeros((0, 0)),
            relations=[r],
            count=r.size(2),
            name=entity2,
            lambda_beta=lambda_beta,
        )

        # Validation for feature matrix dimensions
        if feat1 is not None and feat1.shape[0] != r.size(1):
            raise ValueError(
                f"Number of rows in feat1 ({feat1.shape[0]}) must equal number "
                f"of rows in the relation ({r.size(1)})"
            )

        if feat2 is not None and feat2.shape[0] != Am.shape[1]:
            raise ValueError(
                f"Number of rows in feat2 ({feat2.shape[0]}) must equal number "
                f"of columns in the relation ({Am.shape[1]})"
            )

        # Append entities to the relation
        r.entities.append(e1)
        r.entities.append(e2)

        # Initialize and return RelationData instance with the created entities and relation
        return cls(entities=[e1, e2], relations=[r])

    @classmethod
    def from_dataframe(cls, Am, rname="R1", class_cut=np.log10(200), alpha=5.0):
        """Alternate constructor to initialize RelationData from a DataFrame.

        Parameters:
        Am: pd.DataFrame, DataFrame containing relation data.
        rname: str, Name for the relation.
        class_cut: float, Classification cut-off value.
        alpha: float, Alpha parameter for the relation model.

        Returns:
        RelationData: An instance of RelationData with initialized entities and relation.
        """
        # Calculate maximum values for each column except the last
        dims = [Am.iloc[:, i].max() for i in range(Am.shape[1] - 1)]

        # Initialize an IndexedDF equivalent
        idf = IndexedDF(Am, dims)  # Assuming IndexedDF is a defined class handling indexed data

        # Create an empty RelationData instance
        rd = cls()

        # Initialize and add a Relation to the RelationData's relations list
        relation = Relation(idf, name=rname, class_cut=class_cut, alpha=alpha)
        rd.relations.append(relation)

        # Create and add entities based on the dimensions and column names
        for d in range(len(dims)):
            entity_name = Am.columns[d]  # Get the name of the column
            en = Entity(
                F=np.zeros((0, 0)),
                relations=[relation],
                count=dims[d],
                name=entity_name,
            )

            # Add entity to RelationData's entities list and to the relation's entities list
            rd.entities.append(en)
            rd.relations[0].entities.append(en)

        return rd

    @classmethod
    def from_sparse_matrix(cls, M, **kwargs):
        """Alternate constructor to initialize RelationData from a sparse matrix (CSC
        format).

        Parameters:
        M: scipy.sparse.csc_matrix, Sparse matrix of type float64 and int64 indices.
        **kwargs: Additional keyword arguments for initializing RelationData.

        Returns:
        RelationData: An instance of RelationData with entities and relations derived from the
        sparse matrix.
        """
        # Get the dimensions of the sparse matrix
        dims = M.shape

        # Calculate the column indices for non-zero entries, repeating row index by column size
        col_counts = np.diff(M.indptr)
        cols = rep_int(np.arange(1, dims[1] + 1), col_counts)

        # Extract non-zero row indices, column indices, and values
        row, _, data = find(M)

        # Create a DataFrame from the non-zero entries
        df = pd.DataFrame(
            {"row": row, "col": cols - 1, "value": data}
        )  # Adjust columns to 0-indexed

        # Initialize an IndexedDF with the DataFrame and dimensions
        idf = IndexedDF(df, dims)  # Assuming IndexedDF is a defined class

        # Pass the IndexedDF and other keyword arguments to create a RelationData instance
        return cls(idf, **kwargs)

    @classmethod
    def from_relation(cls, relation):
        """Alternate constructor to initialize RelationData from an existing Relation.

        Parameters:
        relation: Relation, An existing relation to add to the RelationData.

        Returns:
        RelationData: A new instance of RelationData containing the specified relation.
        """
        # Create a new RelationData instance
        rd = cls()

        # Add the provided relation to the RelationData instance
        rd.add_relation(relation)
        return rd

    def add_relation(self, relation):
        """Adds a Relation to the RelationData.

        Parameters:
        relation: Relation, The relation to add to the RelationData.
        """
        self.relations.append(relation)
        # Add entities associated with the relation if not already in the entities list
        for entity in relation.entities:
            if entity not in self.entities:
                self.entities.append(entity)


def rep_int(x, times):
    """Repeats each element in x according to the corresponding value in times.

    Parameters:
    x: np.array, Array of values to be repeated.
    times: list or np.array, List of integers indicating how many times each element in x
    should be repeated.

    Returns:
    np.array: Array with elements of x repeated according to times.
    """
    # Create an output array of zeros with the correct length
    out = np.zeros(sum(times), dtype=x.dtype)

    # Initialize index for tracking position in out
    idx = 0
    for i in range(len(x)):
        # Set the repeated elements in the output array
        out[idx : idx + times[i]] = x[i]
        idx += times[i]

    return out


def F_mul_beta(entity):
    """Computes the matrix product F * beta for an Entity.

    Parameters:
    entity: Entity, The entity containing F and beta.

    Returns:
    np.array: The result of the matrix product F * beta.
    """
    # Check if Frefs is non-empty; if so, use Frefs_mul_B function
    if entity.Frefs:
        return Frefs_mul_B(entity.Frefs, entity.model.beta)
    else:
        # Otherwise, perform the matrix multiplication F * beta
        return np.dot(entity.F, entity.model.beta)


def Ft_mul_B(entity, B):
    """Computes the matrix product F' * B (transpose of F times B) for an Entity.

    Parameters:
    entity: Entity, The entity containing F and Frefs.
    B: np.array, Matrix to multiply with F'.

    Returns:
    np.array: The result of the matrix product F' * B.
    """
    # Check if Frefs is non-empty; if so, use Frefs_t_mul_B function
    if entity.Frefs:
        return Frefs_t_mul_B(entity.Frefs, B)
    else:
        # Otherwise, perform the matrix multiplication of F' (transpose of F) and B
        return np.dot(entity.F.T, B)


def reset(data, num_latent, lambda_beta=np.nan, compute_ff_size=6500, cg_pids=None):
    """Resets the parameters for each entity and relation within a RelationData object.

    Parameters:
    data: RelationData, The RelationData instance containing entities and relations.
    num_latent: int, Number of latent features.
    lambda_beta: float, Optional lambda_beta value for initializing the model (default: NaN).
    compute_ff_size: int, Maximum size to allow direct computation of FF.
    cg_pids: list, List of process IDs for parallel processing.
    """
    if cg_pids is None:
        cg_pids = [myid()]

    # Initialize each entity in the RelationData
    for en in data.entities:
        # Initialize the model for the entity with specified latent features
        init_model(en, num_latent, lambda_beta=lambda_beta)

        # Set up modes and modes_other based on the entity's relations
        en.modes = [next(i for i, ent in enumerate(r.entities) if ent == en) for r in en.relations]
        en.modes_other = [
            [i for i, ent in enumerate(r.entities) if ent != en] for r in en.relations
        ]

        # Handle feature matrix F and its product FF based on compute_ff_size
        if en.has_features():
            if en.F.shape[1] <= compute_ff_size:
                en.FF = np.dot(en.F.T, en.F)  # Equivalent to full(At_mul_B(en.F, en.F))
                en.use_FF = True
            else:
                # Initialize Frefs for CG setup if F is too large
                init_Frefs(en, num_latent, cg_pids)
                en.use_FF = False

    # Initialize each relation in the RelationData
    for r in data.relations:
        # Set mean_value for the relation model
        r.model.mean_value = value_mean(r.data)
        r.temp = RelationTemp()

        # Handle F and FF based on compute_ff_size
        if r.has_features() and r.F.shape[1] <= compute_ff_size:
            r.temp.linear_values = np.full(num_data(r), r.model.mean_value)
            r.temp.FF = np.dot(r.F.T, r.F)  # Equivalent to full(r.F' * r.F)


def init_Frefs(entity, num_latent, pids):
    """Initializes Frefs for an Entity by distributing tasks based on the type of pids.

    Parameters:
    entity: Entity, The entity for which Frefs is initialized.
    num_latent: int, Number of latent features.
    pids: list or dict, Could be:
          - a list of int (list of process IDs for parallel processing),
          - a list of lists (each sublist contains a primary process ID and additional
          associated IDs),
          - or a dictionary where keys are primary process IDs and values are lists of
          associated IDs.
    """
    # Create a non-shared copy of F (assuming nonshared is a function that does this)
    Fns = np.copy(entity.F)

    # Clear any existing Frefs
    entity.Frefs.clear()

    # Use a process pool to spawn tasks for parallel processing
    with ProcessPoolExecutor() as executor:
        # Case 1: pids is a list of integers
        if isinstance(pids, list) and all(isinstance(pid, int) for pid in pids):
            for i in range(min(len(pids), num_latent)):
                pid = pids[i]
                future = executor.submit(copyto, Fns, [pid])
                entity.Frefs.append(future)

        # Case 2: pids is a list of lists (each containing a primary PID and additional IDs)
        elif isinstance(pids, list) and all(isinstance(sublist, list) for sublist in pids):
            for i in range(min(len(pids), num_latent)):
                pid = pids[i][0]
                mulpids = pids[i][1:] if len(pids[i]) > 1 else [pid]
                future = executor.submit(copyto, Fns, mulpids)
                entity.Frefs.append(future)

        # Case 3: pids is a dictionary where keys are primary PIDs and values are lists of additional IDs
        elif isinstance(pids, dict):
            for pid, mulpids in pids.items():
                future = executor.submit(copyto, Fns, mulpids)
                entity.Frefs.append(future)
        else:
            raise TypeError("pids must be a list of integers, a list of lists, " "or a dictionary")


def add_relation(rd, relation):
    """Adds a Relation to the RelationData, validating entities and updating counts as
    needed.

    Parameters:
    rd: RelationData, The RelationData instance to add the relation to.
    relation: Relation, The relation to add.

    Raises:
    ValueError: If the number of entities does not match the data dimensions, or if an entity's count is inconsistent.
    """
    # Check if the number of dimensions in the data matches the number of entities in the relation
    if len(relation.size()) != len(relation.entities):
        raise ValueError(
            f"Relation has {len(relation.entities)} entities, but its data implies {len(relation.size())} dimensions."
        )

    # Add the relation to the RelationData's relations list
    rd.relations.append(relation)

    # Loop through each entity in the relation
    for i, entity in enumerate(relation.entities):
        # Update the entity count if it's currently zero
        if entity.count == 0:
            entity.count = relation.size(i + 1)
        # Check for consistency in entity count
        elif entity.count != relation.size(i + 1):
            raise ValueError(
                f"Entity '{entity.name}' has {entity.count} instances, "
                f"but relation '{relation.name}' has data for {relation.size(i + 1)} instances."
            )

        # Add entity to RelationData if not already present
        if entity not in rd.entities:
            rd.entities.append(entity)

        # Add relation to the entity's relations list if not already present
        if relation not in entity.relations:
            entity.relations.append(relation)


def show(item):
    """
    Displays detailed information based on the type of item:
    RelationData, Relation, or Entity.
    """
    if isinstance(item, RelationData):
        # Display RelationData details
        print("[Relations]")
        for r in item.relations:
            relation_info = (
                f"{r.name:10}: {'--'.join([e.name for e in r.entities])}, "
                f"#known = {num_data(r)}, #test = {num_test(r)}, "
                f"α = {'sample' if r.model.alpha_sample else f'{r.model.alpha:.2f}'}"
            )
            if has_features(r):
                relation_info += f", #feat = {r.F.shape[1]}"
            print(relation_info)

        print("\n[Entities]")
        for en in item.entities:
            entity_info = f"{en.name:10}: {en.count:6d} "
            if has_features(en):
                entity_info += (
                    f"with {en.F.shape[1]} features (λ = "
                    f"{'sample' if en.lambda_beta_sample else f'{en.lambda_beta:.1f}'})"
                )
            else:
                entity_info += "with no features"
            print(entity_info)

    elif isinstance(item, Relation):
        # Display Relation details
        relation_info = (
            f"[Relation] {item.name}: {'--'.join([e.name for e in item.entities])}, "
            f"#known = {num_data(item)}, #test = {num_test(item)}, "
            f"α = {'sample' if item.model.alpha_sample else f'{item.model.alpha:.2f}'}"
        )
        if has_features(item):
            relation_info += f", #feat = {item.F.shape[1]}"
        print(relation_info)

    elif isinstance(item, Entity):
        # Display Entity details
        entity_info = f"[Entity] {item.name}: {item.count:6d} "
        if has_features(item):
            entity_info += (
                f"with {item.F.shape[1]} features (λ = "
                f"{'sample' if item.lambda_beta_sample else f'{item.lambda_beta:.1f}'})"
            )
        else:
            entity_info += "with no features"
        print(entity_info)

    else:
        raise TypeError("Unsupported type. Expected RelationData, Relation, or Entity.")


def normalize_features(entity):
    """Normalizes the columns of the feature matrix F in an Entity.

    Parameters:
    entity: Entity, The entity whose features will be normalized.
    """
    # Compute the square root of the sum of squares for each column (feature)
    diagsq = np.sqrt(np.sum(entity.F**2, axis=0))

    # Normalize each column by scaling with the reciprocal of diagsq
    scaling_matrix = diags(1.0 / diagsq, offsets=0)
    entity.F = entity.F @ scaling_matrix


def normalize_rows(entity):
    """Normalizes the rows of the feature matrix F in an Entity.

    Parameters:
    entity: Entity, The entity whose rows will be normalized.
    """
    # Compute the square root of the sum of squares for each row
    diagf = np.sqrt(np.sum(entity.F**2, axis=1))

    # Normalize each row by scaling with the reciprocal of diagf
    scaling_matrix = diags(1.0 / diagf, offsets=0)
    entity.F = scaling_matrix @ entity.F


def load_mf1c(
    ic50_file="chembl_19_mf1c/chembl-IC50-346targets.csv",
    cmp_feat_file="chembl_19_mf1c/chembl-IC50-compound-feat.csv",
    normalize_feat=False,
    alpha_sample=False,
):
    """Loads data from the IC50 and compound feature files, processes it, and
    initializes RelationData.

    Parameters:
    ic50_file: str, Path to the IC50 data file.
    cmp_feat_file: str, Path to the compound feature file.
    normalize_feat: str or bool, Normalization method ('rows', 'features', or False).
    alpha_sample: bool, Flag for alpha sampling.

    Returns:
    RelationData: An initialized RelationData object.
    """
    # Load IC50 matrix
    X = pd.read_csv(ic50_file)
    X.rename(columns={"row": "compound", "col": "target"}, inplace=True)

    # Calculate dimensions based on maximum values in compound and target columns
    dims = [X["compound"].max(), X["target"].max()]

    # Log-transform the value column
    X["value"] = np.log10(X["value"]) + 1e-5

    # Split off 20% of data for testing
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

    # Load feature matrix
    feat = pd.read_csv(cmp_feat_file)
    F = csr_matrix((np.ones(len(feat)), (feat["compound"], feat["feature"])))

    # Create an IndexedDF and RelationData from X_train
    Xi = IndexedDF(X_train, dims)  # Assumes IndexedDF is defined elsewhere
    data = RelationData(Xi, feat1=F, alpha_sample=alpha_sample)

    # Set test data for the first relation
    data.relations[0].test_vec = X_test
    data.relations[0].test_label = X_test["value"].values < np.log10(200)

    # Normalize features if requested
    if normalize_feat:
        if normalize_feat == "rows":
            normalize_rows(data.entities[0])
        else:
            normalize_features(data.entities[0])

    return data
