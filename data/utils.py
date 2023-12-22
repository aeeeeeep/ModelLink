class IsNoneError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "expect not None variable"


class IsNotValidError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self._error_message = error_message

    def __repr__(self):
        if self._error_message:
            return self._error_message
        else:
            return "Expression is not valid"


def ensure_valid(expression, error_message=None):
    if not expression:
        raise IsNotValidError(error_message)


def ensure_var_is_not_none(variable, error_message=None):
    if variable is not None:
        return
    else:
        raise IsNoneError(error_message=error_message)