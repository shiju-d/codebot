import pytest
from graph import _parse_ruby, _parse_ts_js


# --- _parse_ruby ---

def test_parse_ruby_bare_require():
    source = "require 'snt/channex/exporters/rate_exporter'\n"
    result = _parse_ruby(source, "rover-ifc/test/integration/channex_test.rb", "rover-ifc")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "snt/channex/exporters/rate_exporter"
    assert result.requires[0].is_relative is False


def test_parse_ruby_require_relative():
    source = "require_relative '../exporters/rate_exporter'\n"
    result = _parse_ruby(source, "rover-ifc/lib/snt/channex/services/rate_sync.rb", "rover-ifc")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "../exporters/rate_exporter"
    assert result.requires[0].is_relative is True


def test_parse_ruby_class_and_method():
    source = """
class RateExporter
  def build_occupancy_rates(room_rate_data)
    single_rate = room_rate_data[:single_amount].to_f
  end
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb", "rover-ifc")
    class_names = [c.name for c in result.classes]
    assert "RateExporter" in class_names
    method_names = [m.name for m in result.methods]
    assert "build_occupancy_rates" in method_names


def test_parse_ruby_inheritance():
    source = """
class RateExporter < BaseExporter
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/foo.rb", "rover-ifc")
    assert any(c.name == "RateExporter" and c.parent == "BaseExporter" for c in result.classes)


def test_parse_ruby_include():
    source = """
class Foo
  include RateHelpers
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/foo.rb", "rover-ifc")
    assert any(i.class_name == "Foo" and i.module_name == "RateHelpers" for i in result.includes)


def test_parse_ruby_multiple_requires():
    source = """
require 'foo'
require 'bar'
"""
    result = _parse_ruby(source, "rover-ifc/lib/baz.rb", "rover-ifc")
    require_strs = [r.require_str for r in result.requires]
    assert "foo" in require_strs
    assert "bar" in require_strs


def test_parse_ruby_empty_source():
    result = _parse_ruby("", "rover-ifc/lib/empty.rb", "rover-ifc")
    assert result.requires == []
    assert result.classes == []
    assert result.methods == []


# --- _parse_ts_js ---

def test_parse_ts_relative_import():
    source = "import { RateService } from './rate.service';\n"
    result = _parse_ts_js(source, "ibe-api/src/services/checkout.service.ts", "ibe", "typescript")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "./rate.service"
    assert result.requires[0].is_relative is True


def test_parse_ts_absolute_import_not_marked_relative():
    source = "import { Injectable } from '@angular/core';\n"
    result = _parse_ts_js(source, "ibe-admin/src/app/foo.ts", "ibe", "typescript")
    assert len(result.requires) == 1
    assert result.requires[0].is_relative is False


def test_parse_ts_class_and_method():
    source = """
class CheckoutService {
  async processCheckout(cart: any): Promise<void> {
    return this.validate(cart);
  }
}
"""
    result = _parse_ts_js(source, "ibe-api/src/services/checkout.service.ts", "ibe", "typescript")
    class_names = [c.name for c in result.classes]
    assert "CheckoutService" in class_names
    method_names = [m.name for m in result.methods]
    assert "processCheckout" in method_names


def test_parse_ts_class_inheritance():
    source = """
class CheckoutService extends BaseService {
}
"""
    result = _parse_ts_js(source, "ibe-api/src/foo.ts", "ibe", "typescript")
    assert any(c.name == "CheckoutService" and c.parent == "BaseService" for c in result.classes)


def test_parse_ts_empty_source():
    result = _parse_ts_js("", "ibe-api/src/empty.ts", "ibe", "typescript")
    assert result.requires == []
    assert result.classes == []
    assert result.methods == []
