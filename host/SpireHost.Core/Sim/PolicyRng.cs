namespace SpireHost.Sim;

/// <summary>SplitMix64: tiny, deterministic, independent of the game's RNG streams.</summary>
public sealed class PolicyRng
{
    private ulong _s;

    public PolicyRng(ulong seed)
    {
        _s = seed;
    }

    public ulong Next()
    {
        ulong z = _s += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>Uniform in [0, n) (rejection sampling, no modulo bias).</summary>
    public int NextInt(int n)
    {
        if (n <= 1)
        {
            return 0;
        }
        ulong limit = ulong.MaxValue - (ulong.MaxValue % (ulong)n);
        ulong v;
        do
        {
            v = Next();
        }
        while (v >= limit);
        return (int)(v % (ulong)n);
    }
}
