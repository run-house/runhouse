from typing import Dict, Any

from sphinx.application import Sphinx
from sphinxcontrib.serializinghtml import JSONHTMLBuilder
from sphinx.environment.adapters.toctree import TocTree

__version__ = '0.0.1'

def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(SphinxGlobalTOCJSONHTMLBuilder, override=True)

    return {
        'version': __version__,
        'parallel_read_safe': True
    }

class SphinxGlobalTOCJSONHTMLBuilder(JSONHTMLBuilder):

    name: str = 'json'

    def get_doc_context(self, docname: str, body: str, metatags: str) -> Dict[str, Any]:
        """
        Extends :py:class:`sphinxcontrib.serializinghtml.JSONHTMLBuilder`.

        Add a ``globaltoc`` key to our document that contains the HTML for the
        global table of contents.

        Note:

            We're rendering the **full global toc** for the entire documentation
            set into every page.  We do this so that you can just look at the
            ``master_doc`` and extract its ``globaltoc`` key to get the sitemap
            for the entire set.  Otherwise you'd have to walk through every page
            in the set and merge their individual HTML blobs into a whole.
        """
        doc = super().get_doc_context(docname, body, metatags)
        # Get the entire doctree.  It is the 3rd argument (``collapse``) that
        # does this.  If you set that to ``True`` you will only get the submenu
        # HTML included if you are on a page that is within that submenu.
        self_toctree = TocTree(self.env).get_toctree_for("index", self, False, titles_only=True, includehidden=False, maxdepth=2)
        # self_toctree = global_toctree_for_doc(self.env, "index", self, collapse=False, titles_only=True)
        toctree = self.render_partial(self_toctree)["fragment"]
        doc['globaltoc'] = toctree
        return doc

