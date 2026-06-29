# Suppress pkg_resources deprecation warnings from jieba/pyannote/sphinxcontrib.
# These warnings are emitted at import time and cannot be filtered by mkdocs hooks.
PYTHONWARNINGS := ignore:pkg_resources is deprecated:UserWarning,ignore::DeprecationWarning:pkg_resources

.PHONY: docs
docs:
	PYTHONWARNINGS="$(PYTHONWARNINGS)" mkdocs build

.PHONY: serve
serve:
	PYTHONWARNINGS="$(PYTHONWARNINGS)" mkdocs serve

.PHONY: clean
clean:
	rm -rf site/
