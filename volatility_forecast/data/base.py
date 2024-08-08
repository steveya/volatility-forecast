import pandas as pd
from abc import ABCMeta
from weakref import WeakValueDictionary
from typing import Any, Optional, Union, Mapping, List, Type
from datetime import datetime
from toolz import first

DateLike = Union[datetime, pd.Timestamp, str]


class Node(metaclass=ABCMeta):
    """
    Base class for defining a piece of data.

    This class leverages memoization to cache instances of data with specific parameters.
    It provides utility for avoiding redundant computations and ensuring immutability of data attributes.

    Attributes
    ----------
    dtype : type
        The data type associated with the data (example: float, int).
    params : tuple
        A tuple of parameters associated with the data.
    ndim : int
        The number of dimensions of the data (default is 2).

    Methods
    -------
    _pop_params(kwargs):
        Extracts parameter values from given keyword arguments based on the class's params attribute.
    _static_identity(dtype, ndim, params):
        Computes a unique identity for an instance based on its attributes.
    _init(dtype, ndim, params, *args, **kwargs):
        Initializes the instance with the given attributes.
    _validate():
        Validates that the instance is well-formed. To be overridden by subclasses.
    """

    dtype = None
    params = ()
    ndim = 2
    _data_cache = WeakValueDictionary()

    def __new__(cls, dtype=None, ndim=None, *args, **kwargs):
        """
        Memoized constructor for AbstractData.

        Parameters
        ----------
        dtype : type, optional
            The data type for this instance (default is the class-level dtype attribute).
        ndim : int, optional
            The number of dimensions for this instance (default is the class-level ndim attribute).
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        AbstractData
            A new instance of AbstractData or a cached instance if one with
            equivalent parameters already exists.
        """
        if dtype is None:
            dtype = cls.dtype
        if ndim is None:
            ndim = cls.ndim

        params = cls._pop_params(kwargs)

        identity = cls._static_identity(
            dtype=dtype, ndim=ndim, params=params, *args, **kwargs
        )

        try:
            return cls._data_cache[identity]
        except KeyError:
            new_instance = cls._data_cache[identity] = (
                super(Node, cls)
                .__new__(cls)
                ._init(dtype=dtype, ndim=ndim, params=params, *args, **kwargs)
            )
            return new_instance

    @classmethod
    def _pop_params(cls, kwargs):
        """
        Extracts parameter values from the keyword arguments based on the class's params attribute.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to the class constructor.

        Returns
        -------
        tuple
            A tuple of parameter name and value pairs extracted from kwargs.

        Raises
        ------
        TypeError
            If a required parameter is missing or if a parameter value is not hashable.
        """

        params = cls.params
        if not isinstance(params, Mapping):
            params = {k: None for k in params}
        param_values = []
        for key, default_value in params.items():
            try:
                value = kwargs.pop(key, default_value)
                if value is None:
                    raise KeyError(key)
                hash(value)
            except KeyError:
                raise TypeError(f"{cls.__name__} expected a keyword parameter '{key}'.")
            except TypeError:
                raise TypeError(
                    f"{cls.__name__} expected a hashable value for parameter '{key}', but got '{value}' instead."
                )

            param_values.append((key, value))
        return tuple(param_values)

    def __init__(self, *args, **kwargs):
        """
        Noop constructor to play nicely with our caching __new__.  Subclasses
        should implement _init instead of this method.

        When a class' __new__ returns an instance of that class, Python will
        automatically call __init__ on the object, even if a new object wasn't
        actually constructed.  Because we memoize instances, we often return an
        object that was already initialized from __new__, in which case we
        don't want to call __init__ again.

        Subclasses that need to initialize new instances should override _init,
        which is guaranteed to be called only once.
        """
        pass

    @classmethod
    def _static_identity(cls, dtype, ndim, params):
        """
        Return the identity of the AbstractData that would be constructed from the given arguments.
        given arguments.

        Identities that compare equal will cause us to return a cached instance
        rather than constructing a new one.  We do this primarily because it
        makes dependency resolution easier.

        This is a classmethod so that it can be called from AbstractData.__new__ to
        determine whether to produce a new instance.
        """
        return (cls, dtype, ndim, params)

    def _init(self, dtype, ndim, params):
        """
        Initializes the instance with the given attributes.

        Parameters
        ----------
        dtype : type
            The data type for this instance.
        ndim : int
            The number of dimensions for this instance.
        params : tuple
            A tuple of parameter name and value pairs.

        Returns
        -------
        AbstractData
            The initialized instance.

        Raises
        ------
        TypeError
            If a parameter conflicts with an existing attribute.
        """
        self.dtype = dtype
        self.ndim = ndim
        self.params = dict(params)

        for name, value in params:
            if hasattr(self, name):
                raise TypeError(
                    f"Parameter '{name}' conflicts with existing attribute with value '{getattr(self, name)}'."
                )

        self._subclass_called_super_validate = False
        self._validate()
        assert self._subclass_called_super_validate, (
            "AbstractData._validate() was not called. "
            "Ensure that subclasses call super()."
        )
        del self._subclass_called_super_validate
        return self

    def _validate(self):
        """
        Validates that the instance is well-formed.

        This method should be overridden by subclasses. The base implementation
        sets a flag that ensures subclasses call this method via super().
        """
        self._subclass_called_super_validate = True


class Loadable(Node):
    pass


class Computable(Node):
    """
    A Computable that should be computed from a tuple of inputs.
    """

    inputs = None
    outputs = None

    def __new__(cls, inputs=inputs, outputs=outputs, *args, **kwargs):

        if inputs is None:
            inputs = cls.inputs

        # Having inputs = NotSpecified is an error, but we handle it later
        # in self._validate rather than here.
        if inputs is not None:
            # Allow users to specify lists as class-level defaults, but
            # normalize to a tuple so that inputs is hashable.
            inputs = tuple(inputs)

            # Make sure all our inputs are valid pipeline objects before trying
            # to infer a domain.
            non_terms = [t for t in inputs if not isinstance(t, Node)]
            if non_terms:
                raise ValueError()

        if outputs is None:
            outputs = cls.outputs
        if outputs is not None:
            outputs = tuple(outputs)

        return super(Computable, cls).__new__(
            cls, inputs=inputs, outputs=outputs, *args, **kwargs
        )

    def _init(self, inputs, outputs, *args, **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        return super(Computable, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, inputs, outputs, *args, **kwargs):
        return (
            super(Computable, cls)._static_identity(*args, **kwargs),
            inputs,
            outputs,
        )

    def _validate(self):
        super(Computable, self)._validate()

    def compute(self, universe, start_date, end_date):
        """
        Subclasses should implement this to perform actual computation.
        """
        raise NotImplementedError("compute")

    def __repr__(self):
        return type(self).__name__


class Field:
    """
    An abstract field of data, not yet associated with a dataset.

    This class is used to define the structure of a field in a dataset by specifying its data type,
    documentation, and any additional metadata.

    Attributes
    ----------
    dtype : type
        The data type of the field (e.g., float, int).
    doc : str, optional
        A description or documentation string for the field.
    metadata : dict, optional
        Extra metadata associated with the field.

    Methods
    -------
    bind(name):
        Binds the field to a given name within a dataset, returning a _DataField object.
    """

    def __init__(self, dtype, doc=None, metadata=None):
        self.dtype = dtype
        self.doc = doc
        self.metadata = metadata.copy() if metadata is not None else {}

    def bind(self, name):
        """
        Bind the field to a given name within a dataset.

        Parameters
        ----------
        name : str
            The name to bind to the field.

        Returns
        -------
        _DataField
            An intermediate _DataField object associated with the dataset.
        """
        return _DataField(
            dtype=self.dtype,
            name=name,
            doc=self.doc,
            metadata=self.metadata,
        )


class _DataField(object):
    """
    Intermediate class for associating data fields with datasets.

    This class is used to memoize DataField objects when requested, ensuring that
    datasets don't share columns with their parent classes.

    Attributes
    ----------
    dtype : type
        The data type of the field.
    name : str
        The name of the field within the dataset.
    doc : str
        A description or documentation string for the field.
    metadata : dict
        Extra metadata associated with the field.

    Methods
    -------
    __get__(instance, owner):
        Returns a concrete DataField object when accessed from a dataset.
    """

    def __init__(self, dtype, name, doc, metadata):
        self.dtype = dtype
        self.name = name
        self.doc = doc
        self.metadata = metadata

    def __get__(self, instance, owner):
        """
        Produce a concrete DataField object when accessed.

        This method ensures that subclasses of DataSets produce different DataField objects.

        Parameters
        ----------
        instance : object
            The instance of the dataset.
        owner : type
            The dataset class owning the field.

        Returns
        -------
        DataField
            A concrete DataField object bound to the dataset.
        """
        return DataField(
            dtype=self.dtype,
            dataset=owner,
            name=self.name,
            doc=self.doc,
            metadata=self.metadata,
        )


class DataField(Loadable):
    """
    A data field that has been concretely bound to a particular dataset.

    This class represents a field of data within a dataset, including its data type,
    associated dataset, name, and extra metadata.

    Attributes
    ----------
    dtype : numpy.dtype
        The data type of the data produced when this field is loaded.
    dataset : zipline.pipeline.data.DataSet
        The dataset to which this field is bound.
    name : str
        The name of this field.
    metadata : dict
        Extra metadata associated with this field.

    Methods
    -------
    get_data(dataloader, start_date, end_date):
        Retrieves the data for this field from the given dataloader within the specified date range.
    """

    def __new__(cls, dtype, dataset, name, doc, metadata):
        return super(DataField, cls).__new__(
            cls,
            dtype=dtype,
            dataset=dataset,
            name=name,
            ndim=dataset.ndim,
            doc=doc,
            metadata=metadata,
        )

    def _init(self, dataset, name, doc, metadata, *args, **kwargs):
        """
        Initializes the DataField instance.

        Parameters
        ----------
        dataset : zipline.pipeline.data.DataSet
            The dataset to which this field is bound.
        name : str
            The name of this field.
        doc : str
            A description or documentation string for the field.
        metadata : dict
            Extra metadata associated with this field.
        """
        self._dataset = dataset
        self._name = name
        self.__doc__ = doc
        self._metadata = metadata
        return super(DataField, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, dataset, name, doc, metadata, *args, **kwargs):
        return (
            super(DataField, cls)._static_identity(*args, **kwargs),
            dataset,
            name,
            doc,
            frozenset(sorted(metadata.items(), key=first)),
        )

    @property
    def dataset(self):
        """
        The dataset to which this field is bound.

        Returns
        -------
        dataset : zipline.pipeline.data.DataSet
            The dataset to which this field is bound.
        """
        return self._dataset

    @property
    def name(self):
        """
        The name of this field.

        Returns
        -------
        name : str
            The name of this field.
        """
        return self._name

    @property
    def metadata(self):
        """
        A copy of the metadata for this field.

        Returns
        -------
        dict
            A copy of the metadata associated with this field.
        """
        return self._metadata.copy()

    def get_data(
        self,
        dataloader: "DataLoader",
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> Any:
        """
        Retrieves the data for this field from the given dataloader within the specified date range.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for retrieving data.
        start_date : datetime
            The start date for the data retrieval.
        end_date : datetime
            The end date for the data retrieval.

        Returns
        -------
        data
            The data for this field within the specified date range.
        """

        if not isinstance(dataloader, DataLoader) or dataloader is None:
            raise TypeError("dataloader must be an instance of DataLoader")

        return dataloader.get_data([self], start_date, end_date)[self]

    def __repr__(self):
        """
        Returns a string representation of the DataField.

        Returns
        -------
        str
            A string representation of the DataField.
        """
        return f"{self.name}::{self.dtype}"


class DataSetMeta(type):
    """
    Metaclass for DataSets.

    This metaclass supplies name and dataset information to Field attributes, and manages
    families of specialized datasets.

    Attributes
    ----------
    ndim : int
        The number of dimensions of the dataset (default is 2).

    Methods
    -------
    __new__(mcls, name, bases, dict_):
        Creates a new dataset class, collecting field names from parent classes and adding new fields.
    """

    ndim = 2

    def __new__(mcls, name, bases, dict_):
        if len(bases) > 1:
            raise TypeError("Multiple dataset inheritance is not supported.")

        newtype = super(DataSetMeta, mcls).__new__(mcls, name, bases, dict_)

        # Collect all of the field names that we inherit from our parents.
        field_names = set().union(
            *(getattr(base, "_field_names", ()) for base in bases)
        )

        # Collect any new fields from this dataset.
        for maybe_field_name, maybe_field in dict_.items():
            if isinstance(maybe_field, Field):
                # Add field names defined on our class.
                data_field = maybe_field.bind(maybe_field_name)
                setattr(newtype, maybe_field_name, data_field)
                field_names.add(maybe_field_name)

        newtype._field_names = frozenset(field_names)

        return newtype

    @property
    def fields(self) -> frozenset[DataField]:
        """
        Returns a set of all fields associated with this dataset.

        Returns
        -------
        frozenset
            A set of DataField objects associated with this dataset.
        """
        return frozenset(getattr(self, colname) for colname in self._field_names)

    def __repr__(self) -> str:
        """
        Returns a string representation of the DataSet.

        Returns
        -------
        str
            A string representation of the DataSet.
        """
        return "<DataSet: %r>" % (self.__name__)


def bulleted_list(items, max_count=None, indent=2):
    """Format a bulleted list of values.

    Parameters
    ----------
    items : list
        A list of items to format as a bulleted list.
    max_count : int, optional
        The maximum number of items to display in the list. Additional items will be replaced with '...'.
    indent : int, optional
        The number of spaces to use for indenting each list item.

    Returns
    -------
    str
        A string representation of the bulleted list.
    """
    if max_count is not None and len(items) > max_count:
        item_list = list(items)
        items = item_list[: max_count - 1]
        items.append("...")
        items.append(item_list[-1])

    line_template = (" " * indent) + "- {}"
    return "\n".join(map(line_template.format, items))


class DataSet(metaclass=DataSetMeta):
    """
    Abstract base class for defining datasets.

    This class uses the DataSetMeta metaclass to manage fields of data and their associations
    within the dataset.

    Methods
    -------
    get_field(cls, name):
        Looks up a field by name within the dataset.
    """

    @classmethod
    def get_field(cls, name: str) -> DataField:
        """Look up a column by name.

        Parameters
        ----------
        name : str
            Name of the field to look up.

        Returns
        -------
        DataField
            The DataField object corresponding to the requested name.

        Raises
        ------
        AttributeError
            If no field with the given name exists.
        """
        clsdict = vars(cls)
        try:
            maybe_field = clsdict[name]
            if not isinstance(maybe_field, _DataField):
                raise KeyError(name)
        except KeyError:
            raise AttributeError(
                "{dset} has no field {field_name!r}:\n\n"
                "Possible choices are:\n"
                "{choices}".format(
                    dset=cls.__name__,
                    field_name=name,
                    choices=bulleted_list(
                        sorted(cls._field_names),
                        max_count=10,
                    ),
                )
            )

        # Resolve field descriptor into a DataField.
        return maybe_field.__get__(None, cls)


def ensure_timestamp(date: Optional[DateLike]) -> Optional[pd.Timestamp]:
    """
    Ensure that the input is converted to a pd.Timestamp.

    Parameters
    ----------
    date : str, datetime, pd.Timestamp, or None
        The date to be converted.

    Returns
    -------
    pd.Timestamp or None
    """
    return pd.Timestamp(date) if date is not None else None


class ParametrizedSingleton(ABCMeta):
    """
    Metaclass that creates a singleton instance based on constructor parameters.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super(ParametrizedSingleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[key]


class DataLoader(metaclass=ParametrizedSingleton):
    """
    Abstract base class for data loaders.

    This class defines the interface for loading data fields within a specified date range.

    Methods
    -------
    get_data(self, field, start_date, end_date):
        Retrieves the data for the specified field within the given date range.
    """

    DEFAULT_START_DATE = pd.Timestamp("2000-01-01")
    DEFAULT_END_DATE = pd.Timestamp("today")

    def _get_data(
        self,
        fields: List[DataField],
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> Any:
        """
        Abstract method for fetching the data. To be implemented by subclasses.

        Parameters
        ----------
        field : DataField
            The data field for which data is to be retrieved.
        start_date : pd.Timestamp
            The start date for the data retrieval.
        end_date : pd.Timestamp
            The end date for the data retrieval.

        Returns
        -------
        Any
            The data for the specified field within the date range.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def get_data(
        self,
        fields: List[DataField],
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> Any:
        """
        Retrieves the data for the specified field within the given date range.

        Parameters
        ----------
        field : DataField
            The data field for which data is to be retrieved.
        start_date : str, datetime, or pd.Timestamp, optional
            The start date for the data retrieval. If None, uses the default start date.
        end_date : str, datetime, or pd.Timestamp, optional
            The end date for the data retrieval. If None, uses the default end date.

        Returns
        -------
        Any
            The data for the specified field within the date range.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        start_date = ensure_timestamp(start_date) or self.DEFAULT_START_DATE
        end_date = ensure_timestamp(end_date) or self.DEFAULT_END_DATE

        return self._get_data(fields, start_date, end_date)
