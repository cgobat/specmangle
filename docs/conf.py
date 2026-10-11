from datetime import UTC, datetime
from importlib import metadata

from sphinx_astropy.conf.v3 import *  # noqa: F403
from sphinx_astropy.conf.v3 import extensions, intersphinx_mapping


extensions.append("sphinx_design")

project = "specmangle"
author = "Caden Gobat"
copyright = f"{datetime.now(tz=UTC).year}, {author}"

release = metadata.version("specmangle")
version = release

html_title = f"{project} v{release}"
html_baseurl = "https://specmangle.readthedocs.io/"
html_logo = None
html_favicon = "assets/favicon.png"
html_theme_options = {
    "github_url": "https://github.com/cgobat/specmangle",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "sst_project_name": "specmangle",
    "sst_site_root": "https://specmangle.readthedocs.io/",
    "navbar_links": [],
    "footer_links": [],
    "goatcounter_analytics_url": False,
}
html_context = {
    "github_user": "cgobat",
    "github_repo": "specmangle",
    "github_version": "main",
    "doc_path": "docs/",
}

intersphinx_mapping.update(
    {
        "astropy": ("https://docs.astropy.org/en/stable/", None),
        "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "specutils": ("https://specutils.readthedocs.io/en/stable/", None),
    }
)

numpydoc_xref_param_type = False
modindex_common_prefix = ["specmangle."]
github_issues_url = "https://github.com/cgobat/specmangle/issues/"
edit_on_github_branch = "main"
